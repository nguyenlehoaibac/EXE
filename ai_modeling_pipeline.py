import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def block_bootstrap_time_series(df, block_size=7, num_samples=2):
    print(f"🔄 Đang thực hiện Time-series Bootstrapping...")
    bootstrapped_blocks = []
    n = len(df)
    for _ in range(num_samples):
        for i in range(0, n - block_size + 1, block_size):
            start_idx = np.random.randint(0, n - block_size + 1)
            block = df.iloc[start_idx : start_idx + block_size].copy()
            bootstrapped_blocks.append(block)
    augmented_df = pd.concat(bootstrapped_blocks, ignore_index=True)
    augmented_df['date'] = pd.date_range(start=df['date'].min(), periods=len(augmented_df), freq='D').strftime('%Y-%m-%d')
    return augmented_df

def main_pipeline():
    input_path = os.path.join("data", "raw", "final_features.csv")
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file {input_path}.")
        return
        
    df = pd.read_csv(input_path)
    
    # Đồng bộ tên biến môi trường từ file gốc
    if 'is_extreme_heat' in df.columns:
        df.rename(columns={'is_extreme_heat': 'is_hot_day'}, inplace=True)
    if 'is_hot_day' not in df.columns:
        df['is_hot_day'] = (df['max_temp'] > 33.0).astype(int)

    # ----------------------------------------------------
    # LOGIC MỚI: GIẢ LẬP NHU CẦU ĐA KỊCH BẢN (SỬA LỖI HẰNG SỐ)
    # ----------------------------------------------------
    np.random.seed(42)
    items_config = {
        'SP-001-BIM-BIM': {'base': 35, 'temp_coef': 0.1, 'rain_coef': -0.2, 'lag_1_coef': 0.4, 'lag_7_coef': 0.3},
        'SP-002-BIA-TIGER': {'base': 120, 'temp_coef': 4.5, 'rain_coef': -5.0, 'lag_1_coef': 0.3, 'lag_7_coef': 0.2},
        'SP-003-KEM-CHONG-NANG': {'base': 20, 'temp_coef': 2.5, 'rain_coef': -2.0, 'lag_1_coef': 0.2, 'lag_7_coef': 0.4},
        'SP-004-AO-MUA': {'base': 5, 'temp_coef': -0.4, 'rain_coef': 18.0, 'lag_1_coef': 0.1, 'lag_7_coef': 0.1}
    }
    
    augmented_item_list = []
    holiday_cols = [c for c in df.columns if 'dummy_event' in c or 'event' in c or 'is_holiday' in c]

    for item_id, cfg in items_config.items():
        item_df = df.copy()
        demands = np.zeros(len(item_df))
        
        # Mô phỏng chuỗi tuần tự sinh biến lịch sử thực tế
        for t in range(len(item_df)):
            weather_effect = item_df['max_temp'].iloc[t] * cfg['temp_coef'] + item_df['rain_sum'].iloc[t] * cfg['rain_coef']
            holiday_effect = item_df[holiday_cols].iloc[t].max() * 50 if holiday_cols else 0
            
            l1 = demands[t-1] if t > 0 else cfg['base']
            l7 = demands[t-7] if t > 6 else cfg['base']
            
            val = cfg['base'] + weather_effect + holiday_effect + cfg['lag_1_coef'] * l1 + cfg['lag_7_coef'] * l7
            demands[t] = max(0, val + np.random.normal(0, 8))
            
        item_df['demand'] = demands
        item_df['item_id'] = item_id
        
        # Thực hiện Bootstrapping độc lập bảo toàn chuỗi thời gian của Item
        item_augmented = block_bootstrap_time_series(item_df, block_size=7, num_samples=2)
        
        # Kỹ thuật biến đổi đặc trưng (Feature Engineering) trên từng Item độc lập
        item_augmented['lag_1'] = item_augmented['demand'].shift(1)
        item_augmented['lag_7'] = item_augmented['demand'].shift(7)
        item_augmented['rolling_mean_7'] = item_augmented['demand'].shift(1).rolling(window=7).mean()
        item_augmented['rolling_std_7'] = item_augmented['demand'].shift(1).rolling(window=7).std()
        
        augmented_item_list.append(item_augmented)
        
    df_augmented = pd.concat(augmented_item_list, ignore_index=True)
    df_augmented.dropna(inplace=True)

    # Phân tách tập Train / Val
    split_idx = int(len(df_augmented) * 0.8)
    train_df = df_augmented.iloc[:split_idx]
    val_df = df_augmented.iloc[split_idx:]

    # Prophet Baseline (Gom nhóm theo ngày để đánh giá xu hướng vĩ mô)
    prophet_train = train_df.groupby('date')['demand'].mean().reset_index().rename(columns={'date': 'ds', 'demand': 'y'})
    prophet_val = val_df.groupby('date')['demand'].mean().reset_index().rename(columns={'date': 'ds', 'demand': 'y'})
    m_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m_prophet.fit(prophet_train)
    y_pred_prophet = m_prophet.predict(prophet_val[['ds']])['yhat'].values

    # ----------------------------------------------------
    # ÉP ĐÚNG DANH SÁCH 8 CỘT ĐẦU VÀO KHỚP API 100%
    # ----------------------------------------------------
    feature_cols = ['max_temp', 'rain_sum', 'is_heavy_rain', 'is_hot_day', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
    X_train, y_train = train_df[feature_cols], train_df['demand']
    X_val, y_val = val_df[feature_cols], val_df['demand']
    
    m_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    m_xgb.fit(X_train, y_train)
    y_pred_xgb = m_xgb.predict(X_val)
    
    # Đánh giá chỉ số chất lượng mô hình
    mape_p = calculate_mape(prophet_val['y'], y_pred_prophet)
    mape_x = calculate_mape(y_val, y_pred_xgb)
    winner = "XGBoost" if mape_x < mape_p else "Prophet"
    mae_x = mean_absolute_error(y_val, y_pred_xgb)
    rmse_x = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
    
    print(f"📊 CÁC CHỈ SỐ ĐÁNH GIÁ CỦA XGBOOST VỚI 8 TÍNH NĂNG CHUẨN:")
    print(f"- MAE  (Sai số tuyệt đối): {mae_x:.2f} đơn vị")
    print(f"- RMSE (Độ lệch chuẩn sai số): {rmse_x:.2f} đơn vị")
    print(f"- MAPE (Sai số phần trăm): {mape_x:.2f}%")
    print(f"\n🏆 MÔ HÌNH CHIẾN THẮNG: {winner}")

    # Đóng gói và lưu mô hình
    print("\n💾 --- Đang đóng gói và lưu mô hình tốt nhất ---")
    os.makedirs(os.path.join("data", "models"), exist_ok=True)
    model_path = os.path.join("data", "models", "xgboost_demand_model.joblib")
    joblib.dump(m_xgb, model_path)
    print(f"✅ Đã lưu file mô hình thành công tại: {model_path}")

    # Trực quan hóa kết quả dự báo của 1 sản phẩm tiêu biểu (Bia Tiger)
    print("\n📈 --- Đang xuất biểu đồ dự báo trực quan ---")
    plt.figure(figsize=(14, 6))
    
    val_df['pred_demand'] = y_pred_xgb
    plot_df = val_df[val_df['item_id'] == 'SP-002-BIA-TIGER'].sort_values('date')
    if plot_df.empty:
        plot_df = val_df.sort_values('date')
        
    plt.plot(plot_df['date'].values, plot_df['demand'].values, label='Thực tế (Bia Tiger Actual)', color='#1f77b4', linewidth=2)
    plt.plot(plot_df['date'].values, plot_df['pred_demand'].values, label='AI Dự báo (XGBoost)', color='#ff7f0e', linestyle='--', linewidth=2)
    
    plt.title("SmartSupplyAI: Nhu cầu Thực tế vs Dự báo (Sản phẩm: Bia Tiger)", fontsize=14, fontweight='bold')
    plt.xlabel("Ngày tháng", fontsize=12)
    plt.ylabel("Sản lượng (Đơn vị)", fontsize=12)
    plt.xticks(plot_df['date'].values[::7], rotation=45) 
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    chart_path = os.path.join("data", "models", "forecast_chart.png")
    plt.savefig(chart_path)
    print(f"✅ Đã lưu hình ảnh biểu đồ tại: {chart_path}")
    plt.show()

if __name__ == "__main__":
    main_pipeline()