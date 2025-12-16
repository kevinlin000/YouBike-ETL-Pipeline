FROM python:3.10-slim

WORKDIR /app

# 安裝系統編譯工具 (避免有些套件安裝失敗)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. 複製並安裝 requirements (使用我們剛建好的 app 專用清單)
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# 2. 這裡我們「不」用 COPY 指令複製程式碼
# 因為我們在 docker-compose.yaml 裡會用 volumes 掛載整個目錄
# 這樣你修改程式碼後，不需要重新 Build Image 就能生效！