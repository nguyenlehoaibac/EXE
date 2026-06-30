import os
import requests
import pandas as pd

def crawl_vietnam_holidays(year=2026):
    """Sử dụng API Nager.Date để lấy danh sách ngày nghỉ lễ của Việt Nam"""
    print(f"⏳ Đang cào dữ liệu Lễ Tết Việt Nam năm {year}...")
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/VN"
    
    response = requests.get(url)
    if response.status_code == 200:
        holidays_data = response.json()
        # Chuyển đổi JSON thành DataFrame chỉ lấy ngày và tên ngày lễ
        holidays_df = pd.DataFrame(holidays_data)[['date', 'localName', 'name']]
        holidays_df.rename(columns={'name': 'event_type'}, inplace=True)
        return holidays_df
    else:
        print(f"❌ Lỗi khi cào dữ liệu: {response.status_code}")
        return pd.DataFrame(columns=['date', 'localName', 'event_type'])

def build_calendar_with_dummies(year=2026):
    # 1. Tạo bộ khung 365 ngày cho cả năm
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    date_range = pd.date_range(start=start_date, end=end_date)
    
    calendar_df = pd.DataFrame({
        'calendar_date': date_range.strftime('%Y-%m-%d'),
        'day_of_week': date_range.dayofweek, # 0 = Thứ 2, 6 = Chủ nhật
        'is_weekend': (date_range.dayofweek >= 5).astype(int) # Tạo Dummy cho cuối tuần
    })
    
    # 2. Cào dữ liệu lễ tết thực tế
    holidays_df = crawl_vietnam_holidays(year)
    
    # 3. Mapping (Ghép) dữ liệu lễ tết vào bộ khung 365 ngày
    calendar_df = pd.merge(calendar_df, holidays_df[['date', 'event_type']], 
                           left_on='calendar_date', right_on='date', how='left')
    calendar_df.drop(columns=['date'], inplace=True)
    
    # Cột nào không có sự kiện (NaN) thì điền "Normal Day"
    calendar_df['event_type'] = calendar_df['event_type'].fillna('Normal Day')
    
    # Tạo biến Dummy: 1 nếu là ngày lễ, 0 nếu không phải
    calendar_df['is_holiday'] = (calendar_df['event_type'] != 'Normal Day').astype(int)
    
    # 4. TẠO BIẾN DUMMY (One-Hot Encoding) CHO TỪNG LOẠI SỰ KIỆN
    calendar_with_dummies = pd.get_dummies(calendar_df, columns=['event_type'], prefix='event')
    
    # Chuyển đổi các cột dummy True/False thành số 0/1 cho AI dễ học
    for col in calendar_with_dummies.columns:
        if calendar_with_dummies[col].dtype == bool:
            calendar_with_dummies[col] = calendar_with_dummies[col].astype(int)
            
    # 5. Lưu lại thành file CSV đồng bộ cấu hình hệ thống
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    output_path = os.path.join("data", "raw", f"dim_calendar_vietnam_{year}.csv")
    calendar_with_dummies.to_csv(output_path, index=False)
    print(f"🎉 Đã tạo và lưu File Lịch + Biến Dummy siêu chuẩn tại: {output_path}")

if __name__ == "__main__":
    build_calendar_with_dummies(2026)