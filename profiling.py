import os
import pandas as pd
from ydata_profiling import ProfileReport

def run_profiling():
    # Sử dụng file tính năng tổng hợp trong data/raw
    data_path = os.path.join("data", "raw", "final_features.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {data_path}. Vui lòng chạy script thời tiết trước!")
        return

    df = pd.read_csv(data_path)
    print("⏳ Đang phân tích dữ liệu và tạo báo cáo trực quan với YData Profiling...")
    
    # Tạo báo cáo khám phá dữ liệu (EDA) bao gồm Outliers, Correlatons, Missing values
    profile = ProfileReport(df, title="SmartSupplyAI - Data Profiling Report", explorative=True)
    
    output_path = os.path.join("data", "raw", "data_profiling_report.html")
    profile.to_file(output_path)
    print(f"✅ Đã xuất báo cáo thành công! Xem tại: {output_path}")

if __name__ == "__main__":
    run_profiling()