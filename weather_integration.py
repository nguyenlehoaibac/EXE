import os
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather_2026(lat=10.823, lon=106.6296):
    """Gọi API Open-Meteo lấy dữ liệu lịch sử thời tiết năm 2026 cho TP.HCM"""
    safe_end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    print(f"⏳ Đang kết nối API Open-Meteo lấy dữ liệu thời tiết từ 2026-01-01 đến {safe_end_date}...")
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2026-01-01&end_date={safe_end_date}&daily=temperature_2m_max,rain_sum&timezone=Asia%2FBangkok"
    
    response = requests.get(url)
    if response.status_code == 200:
        daily_data = response.json()['daily']
        return pd.DataFrame({
            'date': daily_data['time'],
            'max_temp': daily_data['temperature_2m_max'],
            'rain_sum': daily_data['rain_sum']
        })
    else:
        print(f"❌ Lỗi API ({response.status_code}): {response.text}")
        return None

def build_integrated_features():
    weather_df = fetch_weather_2026()
    if weather_df is None: return

    calendar_path = os.path.join("data", "raw", "dim_calendar_vietnam_2026.csv")
    if not os.path.exists(calendar_path):
        print(f"❌ Thiếu file lịch gốc tại: {calendar_path}")
        return
    calendar_df = pd.read_csv(calendar_path)

    # Giữ lại toàn bộ 365 ngày của file Calendar
    integrated_df = pd.merge(weather_df, calendar_df, left_on='date', right_on='calendar_date', how='right')

    # Xử lý các biến chỉ báo môi trường (Đồng bộ tên biến thành is_hot_day)
    integrated_df['is_heavy_rain'] = (integrated_df['rain_sum'] > 15.0).astype(int).fillna(0)
    integrated_df['is_hot_day'] = (integrated_df['max_temp'] > 33.0).astype(int).fillna(0)
    
    integrated_df = pd.get_dummies(integrated_df, columns=['event_type'], prefix='dummy_event', dtype=int)

    # Dọn dẹp cột thừa
    integrated_df.drop(columns=['calendar_date'], inplace=True)

    output_path = os.path.join("data", "raw", "final_features.csv")
    integrated_df.to_csv(output_path, index=False)
    print(f"📦 Đã tạo xong bộ dữ liệu tích hợp đa chiều tại: {output_path}")

if __name__ == "__main__":
    build_integrated_features()