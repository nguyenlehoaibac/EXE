noi_dung = """# 🚀 Smart Supply AI System

Hệ thống dự báo nhu cầu thông minh tích hợp thời tiết thực tế (OpenWeatherMap), lịch âm dương Việt Nam, và mô hình XGBoost để dự báo demand theo ngày.

## 📑 Mục Lục
1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Kiến Trúc Hệ Thống](#2-kiến-trúc-hệ-thống)
3. [Cấu Trúc Thư Mục](#3-cấu-trúc-thư-mục)
4. [Yêu Cầu Hệ Thống](#4-yêu-cầu-hệ-thống)
5. [Cài Đặt & Khởi Chạy](#5-cài-đặt--khởi-chạy)
6. [Biến Môi Trường](#6-biến-môi-trường)
7. [API Documentation](#7-api-documentation)
8. [Mô Hình AI (XGBoost)](#8-mô-hình-ai-xgboost)
9. [Module Thời Tiết](#9-module-thời-tiết)
10. [Module Lịch Việt Nam](#10-module-lịch-việt-nam)
11. [Logic Phân Loại Thời Tiết](#11-logic-phân-loại-thời-tiết)
12. [Chế Độ Demo (Không Có API Key)](#12-chế-độ-demo-không-có-api-key)
13. [Lỗi Thường Gặp & Cách Xử Lý](#13-lỗi-thường-gặp--cách-xử-lý)
14. [Roadmap & TODO](#14-roadmap--todo)
15. [Ghi Chú Kỹ Thuật Quan Trọng](#15-ghi-chú-kỹ-thuật-quan-trọng)

---

## 1. Tổng Quan Dự Án
### Bài Toán
Dự báo nhu cầu hàng ngày (demand forecasting) cho chuỗi cung ứng tại TP. Hồ Chí Minh, có tính đến:
* **Yếu tố thời tiết thực tế:** Nhiệt độ, mưa, mây - lấy từ OpenWeatherMap API.
* **Yếu tố lịch đặc thù Việt Nam:** Tết Nguyên Đán, ngày lễ âm lịch, ngày nghỉ cuối tuần.
* **Dữ liệu lag chuỗi thời gian:** Nhu cầu ngày hôm qua (`lag_1`), 7 ngày trước (`lag_7`), trung bình và độ lệch chuẩn 7 ngày qua.

### Đầu Vào / Đầu Ra
| Thành phần | Mô Tả |
| :--- | :--- |
| **Input** | 4 giá trị time-series do người dùng cung cấp + thời tiết & lịch tự động lấy. |
| **Output** | Số lượng demand dự báo (integer) + toàn bộ thông tin thời tiết & lịch chi tiết. |

### Stack Công Nghệ
| Thành phần | Công nghệ |
| :--- | :--- |
| **Web Framework** | FastAPI |
| **ML Model** | XGBoost (lưu dưới dạng joblib) |
| **Data Processing** | Pandas |
| **Weather API** | OpenWeatherMap (endpoint: `/data/2.5/weather`) |
| **Runtime** | Python 3.10+ |

---

## 2. Kiến Trúc Hệ Thống
Hệ thống vận hành theo luồng xử lý đồng bộ theo thời gian thực (Real-time Pipeline):
1. **Client** gửi một yêu cầu HTTP POST kèm dữ liệu chuỗi thời gian (`lag`).
2. **FastAPI Application** tiếp nhận request và kích hoạt song song hai luồng xử lý nội bộ:
   * Gọi hàm `get_live_weather()` để lấy thông tin thời tiết thực tế từ API ngoài.
   * Gọi hàm `get_calendar_info()` để tính toán ngày lễ và lịch âm dương Việt Nam.
3. Dữ liệu từ Client và dữ liệu môi trường (Thời tiết + Lịch) được gộp lại, chuyển hóa thành một cấu trúc mảng đồng nhất (Pandas DataFrame).
4. Gọi hàm `predict()` của mô hình **XGBoost** để đưa ra kết quả dự báo cuối cùng.
5. Trả về cho Client một chuỗi định dạng **JSON** hoàn chỉnh chứa cả số lượng dự báo lẫn thông tin dashboard hiển thị.

---

## 3. Cấu Trúc Thư Mục
`\u0060\u0060\u0060text
smart-supply-ai/
├── data/
│   └── models/
│       └── xgboost_demand_model.joblib  <-- File "bộ não" AI sau khi train
├── main.py                              <-- File mã nguồn xử lý FastAPI Application
├── requirements.txt                     <-- File định nghĩa các thư viện phụ thuộc
└── README.md                            <-- Tài liệu hướng dẫn hệ thống
\u0060\u0060\u0060`
⚠️ **Lưu ý quan trọng:** File mô hình `xgboost_demand_model.joblib` bắt buộc phải được đặt chính xác tại đường dẫn thư mục `data/models/`. Nếu thiếu file này, hệ thống server sẽ lập tức bị crash ngay khi vừa khởi động.

---

## 4. Yêu Cầu Hệ Thống
Hệ thống yêu cầu máy chủ cài đặt môi trường chạy **Python với phiên bản từ 3.10 trở lên**. Các thư viện phụ thuộc lõi và phiên bản khuyến nghị bao gồm:
* `fastapi>=0.110.0`
* `uvicorn>=0.28.0`
* `pandas>=2.0.0`
* `xgboost>=2.0.0`
* `joblib>=1.3.0`
* `requests>=2.31.0`

---

## 5. Cài Đặt & Khởi Chạy
Để cài đặt và phân phối ứng dụng ở môi trường Local (máy cá nhân) hoặc Server, hãy thực hiện tuần tự:

**Bước 1 - Clone mã nguồn từ kho lưu trữ:**
`\u0060\u0060\u0060bash
git clone https://github.com/your-username/smart-supply-ai.git
cd smart-supply-ai
\u0060\u0060\u0060`

**Bước 2 - Tạo và kích hoạt môi trường ảo (Virtual Environment):**
`\u0060\u0060\u0060bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\\Scripts\\activate     # Windows
\u0060\u0060\u0060`

**Bước 3 - Cài đặt thư viện:**
`\u0060\u0060\u0060bash
pip install -r requirements.txt
\u0060\u0060\u0060`

**Bước 4 - Khởi chạy Server:**
`\u0060\u0060\u0060bash
uvicorn main:app --host 0.0.0.0 --port 8000
\u0060\u0060\u0060`

---

## 6. Biến Môi Trường
| Tên biến | Bắt buộc | Mô tả | Mặc định |
| :--- | :--- | :--- | :--- |
| `OPENWEATHER_API_KEY` | Không | Mã khóa xác thực API do OpenWeatherMap cấp. | Nếu bỏ trống, hệ thống tự động chạy **Chế độ Demo**. |

---

## 7. API Documentation
### `POST /predict/smart`
Endpoint chính phục vụ dự báo nhu cầu.

**Request Body JSON:**
`\u0060\u0060\u0060json
{
  "lag_1": 120.5,
  "lag_7": 98.0,
  "rolling_mean_7": 110.3,
  "rolling_std_7": 15.2
}
\u0060\u0060\u0060`

**Response JSON:**
`\u0060\u0060\u0060json
{
  "status": "success",
  "realtime_weather_dashboard": {
    "current_temperature_celsius": 34.5,
    "weather_status": "☀️ Nắng Rải Rác"
  },
  "calendar_dashboard": {
    "day_type": "Ngày Thường",
    "holiday_name": "Ngày Bình Thường"
  },
  "predicted_demand": 134
}
\u0060\u0060\u0060`

---

## 8. Mô Hình AI (XGBoost)
Mô hình yêu cầu DataFrame đầu vào tuân thủ **chính xác 8 tính năng theo thứ tự**:
1. `max_temp` (°C)
2. `rain_sum` (mm)
3. `is_heavy_rain` (1/0)
4. `is_hot_day` (1/0)
5. `lag_1`
6. `lag_7`
7. `rolling_mean_7`
8. `rolling_std_7`
*(Truyền sai thứ tự sẽ làm hỏng kết quả dự báo mà không báo lỗi)*.

---

## 9. Module Thời Tiết
Hệ thống bóc tách API OpenWeatherMap:
* `main.temp` -> `max_temp`
* `main.humidity` -> `humidity`
* `main.pressure` -> `pressure`
* `rain.1h` -> `rain_sum`
* `clouds.all` -> `clouds`
* `weather[0].main` -> `owm_condition`

---

## 10. Module Lịch Việt Nam
Sử dụng thuật toán nội bộ bằng module `datetime` để nhận diện (không phụ thuộc API ngoài):
* **Lễ Dương Lịch:** 01/01, 30/04, 01/05, 02/09, 25/12.
* **Lễ Âm Lịch:** Tết Nguyên Đán, Giỗ Tổ (10/3 AL), Lễ Phật Đản (15/4 AL), Trung Thu (15/8 AL).

---

## 11. Logic Phân Loại Thời Tiết
Hệ thống sử dụng bộ lọc ma trận điều kiện của hàm `classify_weather` để trả về giao diện MSN Weather:
* Nhóm có mưa: `🌦 Mưa Nhỏ`, `🌧 Mưa Vừa`, `🌧 Mưa To`.
* Nhóm không mưa: Dựa vào nhiệt độ và độ che phủ mây để gán `☀️ Nắng Gắt`, `🌤 Nắng Đẹp` hoặc `☁️ Nhiều Mây`.

---

## 12. Chế Độ Demo (Không Có API Key)
Khi biến `OPENWEATHER_API_KEY` bị trống, hệ thống không báo lỗi mà tự động kích hoạt **Mock Data** (`temp = 32.5°C`, `rain = 0.0mm`). Đảm bảo API luôn trả về HTTP 200 OK để Team Web tiếp tục test giao diện.

---

## 13. Lỗi Thường Gặp & Cách Xử Lý
* **Lỗi Crash khi gõ uvicorn:** Thiếu thư mục `data/models/` hoặc file `.joblib`.
* **Lỗi 502 Bad Gateway:** API Key OpenWeatherMap hết hạn hoặc nhập sai.
* **Kết quả dự báo phi logic:** Truyền sai thứ tự 8 tính năng của Pandas DataFrame.

---

## 14. Roadmap & TODO
* Tích hợp PostgreSQL/MongoDB lưu trữ lịch sử dự báo.
* Xây dựng Dashboard Frontend (React.js/Vue.js).
* Tự động Auto-retrain mô hình theo Quý.

---

## 15. Ghi Chú Kỹ Thuật Quan Trọng
Mô hình XGBoost trả về số thực thập phân (`float`). File `main.py` đã tự động bọc bằng hàm `int(round())` để kết quả `predicted_demand` đến tay Client luôn là số nguyên tuyệt đối, hỗ trợ chuẩn xác cho công tác quản lý kho bãi.
"""

# Tự động tạo và ghi nội dung ra file README.md
with open("README.md", "w", encoding="utf-8") as file:
    file.write(noi_dung)

print("✅ Đã tạo file README.md thành công! Bạn hãy kiểm tra lại cột file bên trái của VS Code nhé.")