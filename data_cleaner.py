import os
import pandas as pd
import numpy as np

def clean_and_handle_outliers(file_name="final_features.csv"):
    file_path = os.path.join("data", "raw", file_name)
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    
    # 1. Xử lý Missing Values (Điền giá trị bằng phương pháp nội suy chuỗi thời gian)
    # Nếu có cột số liệu bị trống, tự động điền bằng giá trị của ngày sát trước/sau nó
    df = df.ffill().bfill()
    
    # 2. Xử lý Outliers cho các cột số liệu liên tục (Ví dụ: cột lượng mưa 'rain_sum' hoặc nhiệt độ 'max_temp')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Bỏ qua các cột dạng nhị phân/dummy (chỉ chứa 0 và 1)
        if df[col].nunique() <= 2:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Áp dụng kĩ thuật Clipping đưa giá trị dị biệt về khoảng an toàn mà không làm mất dòng dữ liệu lịch sử
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        print(f"✨ Đã xử lý xong Outliers cho cột [{col}] với biên IQR: ({lower_bound:.2f} -> {upper_bound:.2f})")
        
    cleaned_path = os.path.join("data", "raw", "cleaned_features.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"🚀 Dữ liệu sạch hoàn toàn đã được lưu tại: {cleaned_path}")

if __name__ == "__main__":
    clean_and_handle_outliers()