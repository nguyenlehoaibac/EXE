# 1. Sử dụng base image Python 3.10 siêu nhẹ
FROM python:3.10-slim

# 2. Thiết lập thư mục làm việc trong Container
WORKDIR /app

# 3. Copy file requirements và cài đặt thư viện trước
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. CHỈ COPY ĐÚNG NHỮNG FILE CẦN THIẾT CHO AI SERVICE
# Copy file chạy API chính
COPY main.py . 

# Copy đúng file model đã được huấn luyện vào đúng cấu trúc thư mục
COPY data/models/xgboost_demand_model.joblib ./data/models/

# 5. Mở port 8000 cho FastAPI
EXPOSE 8000

# (Các dòng trên giữ nguyên)
# Lệnh khởi chạy Uvicorn server thông qua Python module
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]