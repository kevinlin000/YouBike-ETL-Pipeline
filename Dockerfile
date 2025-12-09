FROM apache/airflow:2.8.1

# 切換到 root 安裝系統套件 (如果有需要的話，目前先不用)
USER root

# 切換回 airflow 使用者來安裝 Python 套件
USER airflow

# 安裝我們需要的資料工程套件
RUN pip install --no-cache-dir \
    pandas \
    requests \
    sqlalchemy \
    pymysql \
    cryptography \
    google-cloud-secret-manager