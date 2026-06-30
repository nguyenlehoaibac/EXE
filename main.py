from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

app = FastAPI(title="Smart Supply AI System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Tải mô hình AI ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("data", "models", "xgboost_demand_model.joblib")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("⚠️ Cảnh báo: Chưa tìm thấy file mô hình. Hãy chạy ai_modeling_pipeline.py trước!")
    model = None
load_dotenv()  # <-- 2. THÊM DÒNG NÀY (Để mồi hệ thống đọc file .env)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ── Schema đầu vào nâng cấp (Sửa lỗi số 3: Thêm biến location) ────────────────
class ItemInRequest(BaseModel):
    item_id: str
    lag_1: float
    lag_7: float
    rolling_mean_7: float
    rolling_std_7: float

class BatchPredictRequest(BaseModel):
    location: str  # Tên tỉnh/thành phố tiếng Anh chuẩn OpenWeatherMap (Ví dụ: "Hanoi", "Haiphong", "Can Tho")
    items: List[ItemInRequest]

# ── Quản lý Lịch lễ & Tết 2026 ──────────────────────────────────────────────
FIXED_HOLIDAYS: dict[str, str] = {
    "01-01": "Tết Dương Lịch",
    "04-30": "Ngày Giải Phóng Miền Nam",
    "05-01": "Quốc Tế Lao Động",
    "09-02": "Quốc Khánh Việt Nam",
    "12-25": "Lễ Giáng Sinh",
}

LUNAR_HOLIDAYS: dict[str, str] = {
    "2026-02-16": "Giao Thừa Tết Bính Ngọ",
    "2026-02-17": "Mùng 1 Tết",
    "2026-02-18": "Mùng 2 Tết",
    "2026-02-19": "Mùng 3 Tết",
    "2026-02-20": "Mùng 4 Tết",
    "2026-02-21": "Mùng 5 Tết",
    "2026-04-21": "Giỗ Tổ Hùng Vương (10/3 ÂL)",
    "2026-09-25": "Tết Trung Thu (15/8 ÂL)",
}

DAY_OF_WEEK_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

def get_calendar_info(target_date: datetime) -> dict:
    today_str   = target_date.strftime("%Y-%m-%d")
    month_day   = target_date.strftime("%m-%d")
    day_of_week = target_date.weekday()
    is_weekend  = 1 if day_of_week >= 5 else 0

    if today_str in LUNAR_HOLIDAYS:
        is_holiday, holiday_name, day_type = 1, LUNAR_HOLIDAYS[today_str], "Ngày Lễ / Tết"
    elif month_day in FIXED_HOLIDAYS:
        is_holiday, holiday_name, day_type = 1, FIXED_HOLIDAYS[month_day], "Ngày Lễ"
    elif is_weekend:
        is_holiday, holiday_name, day_type = 0, "Ngày Cuối Tuần", "Cuối Tuần"
    else:
        is_holiday, holiday_name, day_type = 0, "Ngày Bình Thường", "Ngày Thường"

    return {
        "current_date":  today_str,
        "day_of_week":   DAY_OF_WEEK_VI[day_of_week],
        "is_weekend":    is_weekend,
        "is_holiday":    is_holiday,
        "holiday_name":  holiday_name,
        "day_type":      day_type,
    }

def classify_weather(owm_condition: str, temp: float, rain_sum: float, clouds: int, description: str = "") -> str:
    cond, desc = owm_condition.lower(), description.lower()
    if cond == "thunderstorm": return "⛈ Giông Bão / Sấm Sét"
    if cond == "drizzle": return "🌦 Mưa Phùn / Mưa Nhỏ"
    if cond == "rain":
        return "🌧 Mưa To / Có Thể Ngập" if rain_sum >= 15 else ("🌧 Mưa Vừa" if rain_sum >= 5 else "🌦 Mưa Nhỏ")
    if cond in ("mist", "fog", "haze", "smoke", "dust", "sand", "ash"): return "🌫 Sương Mù / Tầm Nhìn Kém"
    if cond == "clear":
        return "☀️ Nắng Gắt / Nhiệt Độ Cao" if temp >= 35 else "🌤 Trời Trong / Mát Mẻ"
    if cond == "clouds":
        if "broken clouds" in desc or "overcast clouds" in desc or clouds >= 85: return "🌥 Mây Rải Rác / Ít Nắng"
        return "⛅ Nắng Rải Rác / Mây Rải Rác"
    return "🌡 Thời Tiết Khác"

# ── Sửa lỗi số 2 & 3: Chuyển sang gọi API dự báo tương lai & truyền địa điểm động ──
def get_tomorrow_forecast(location: str) -> dict:
    # Lấy thời gian của NGÀY MAI
    tomorrow = datetime.now() + timedelta(days=1)
    calendar = get_calendar_info(tomorrow)
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")

    # Cấu hình giá trị mặc định phòng trường hợp API Key lỗi
    max_temp, rain_sum, clouds, owm_condition, owm_description = 32.3, 0.0, 80, "Clouds", "broken clouds"

    if OPENWEATHER_API_KEY:
        try:
            # SỬ DỤNG API FORECAST (DỰ BÁO 5 NGÀY / 3 GIỜ) MIỄN PHÍ CỦA OPENWEATHERMAP
            url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={OPENWEATHER_API_KEY}&units=metric&lang=vi"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                
                # Lọc ra tất cả các khung giờ thuộc về NGÀY MAI
                tomorrow_slots = [slot for slot in data.get("list", []) if slot.get("dt_txt", "").startswith(tomorrow_str)]
                
                if tomorrow_slots:
                    # 1. Tìm nhiệt độ lớn nhất (Max Temp) trong các khung giờ ngày mai
                    max_temp = max([slot["main"]["temp_max"] for slot in tomorrow_slots])
                    
                    # 2. Cộng dồn lượng mưa (Rain Sum) dự kiến của tất cả các khung giờ ngày mai
                    rain_sum = sum([slot.get("rain", {}).get("3h", 0.0) for slot in tomorrow_slots])
                    
                    # 3. Lấy thông số mây/trạng thái lúc giữa trưa (12:00 hoặc 15:00) làm đại diện cho cả ngày
                    midday_slot = next((slot for slot in tomorrow_slots if "12:00" in slot["dt_txt"] or "15:00" in slot["dt_txt"]), tomorrow_slots[0])
                    clouds = midday_slot.get("clouds", {}).get("all", 0)
                    owm_condition = midday_slot["weather"][0]["main"]
                    owm_description = midday_slot["weather"][0]["description"]
            else:
                print(f"⚠️ LỖI TỪ OPENWEATHERMAP: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"⚠️ Lỗi kết nối mạng khi gọi API Forecast: {e}")

    weather_status = classify_weather(owm_condition, max_temp, rain_sum, clouds, owm_description)
    is_hot_day = 1 if max_temp >= 33 else 0
    is_heavy_rain = 1 if rain_sum >= 15 else 0

    return {
        "max_temp": max_temp,
        "rain_sum": rain_sum,
        "is_heavy_rain": is_heavy_rain,
        "is_hot_day": is_hot_day,
        "weather_status": weather_status,
        "calendar": calendar,
    }

# ── Endpoint Batch Processing dự báo cho Ngày Mai ──────────────────────────────
@app.post("/predict/smart")
def predict_demand_smart_batch(request: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình AI chưa được tải thành công trên Server.")
    try:
        # Lấy thời tiết dự báo ngày mai dựa vào location do khách hàng gửi lên
        weather = get_tomorrow_forecast(request.location)
        cal = weather["calendar"]

        rows = []
        item_ids = []
        for item in request.items:
            item_ids.append(item.item_id)
            rows.append([
                weather["max_temp"],
                weather["rain_sum"],
                weather["is_heavy_rain"],
                weather["is_hot_day"],
                item.lag_1,
                item.lag_7,
                item.rolling_mean_7,
                item.rolling_std_7,
            ])

        input_df = pd.DataFrame(rows, columns=[
            "max_temp", "rain_sum", "is_heavy_rain", "is_hot_day",
            "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7",
        ])
        
        batch_predictions = model.predict(input_df) 

        predictions_output = []
        for idx, item_id in enumerate(item_ids):
            # SỬA LỖI SỐ 1: KHÔNG LÀM TRÒN THÀNH SỐ NGUYÊN (INT) NỮA
            # Giữ số thực Float và làm tròn 2 chữ số thập phân cho đẹp mắt
            pred_value = round(float(batch_predictions[idx]), 2)
            predictions_output.append({
                "item_id": item_id,
                "predicted_demand": max(0.0, pred_value)
            })

        return {
            "status": "success",
            "context": {
                "predict_for_date": cal["current_date"],  # Trả về ngày mai để Web hiển thị
                "location": request.location,             # Trả về địa điểm đang dự báo
                "weather_forecast": {
                    "max_temperature": round(weather["max_temp"], 1),
                    "total_rain_mm": round(weather["rain_sum"], 1),
                    "condition": weather["weather_status"]
                },
                "calendar": {
                    "day_of_week": cal["day_of_week"],
                    "day_type": cal["day_type"],
                    "holiday_name": cal["holiday_name"]
                }
            },
            "predictions": predictions_output
        }
    except Exception as e:
        return {"status": "error", "message": f"Internal Server Error: {str(e)}"}