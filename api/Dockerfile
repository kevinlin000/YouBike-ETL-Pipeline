FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 1. 先複製 requirements.txt 並安裝套件 (利用 Docker Cache 加速)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 複製 程式碼 (app) 和 模型檔案 (model_files)
COPY app/ ./app/
COPY model_files/ ./model_files/

# 開放 8000 port
EXPOSE 8000

# 啟動 FastAPI 伺服器
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]