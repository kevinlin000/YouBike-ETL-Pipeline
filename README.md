🚲 Taipei YouBike Traffic Prediction System (GCP Cloud-Native)
📌 Project Overview
This project is an End-to-End Data Engineering & Machine Learning pipeline designed to forecast real-time traffic flow for Taipei's YouBike 2.0 system.

The infrastructure is hosted on Google Cloud Platform (GCP) Compute Engine, utilizing Docker for containerization and Apache Airflow for orchestration. A key feature of this system is its focus on Security (via GCP Secret Manager) and Multivariate Analysis (integrating weather data into LSTM models).

🏗️ System Architecture
The system follows a hybrid cloud architecture separating the Production ETL Environment (GCP) from the Analytics Environment (Local).

(Please refer to the System_Architecture_Diagram.jpg in the repo)

1. Cloud Infrastructure (GCP Compute Engine)

Orchestration: Apache Airflow runs scheduled DAGs (Crontab: Every 1 min) to trigger data ingestion.

Security: GCP Secret Manager is integrated to securely retrieve database credentials (mysql_password) at runtime. No sensitive keys are hardcoded.

Storage: MySQL (Dockerized) serves as the central Data Warehouse, storing historical traffic and station metadata.

2. Local Analytics Environment

Data Science: Jupyter Notebooks connect to the Data Warehouse for feature engineering.

Deep Learning: A PyTorch LSTM model is trained locally using GPU acceleration to predict bike availability.

💾 Database Schema Design (MySQL)
The database is designed with a normalized relational schema to optimize storage efficiency.

station_info (Dimension Table):

Stores static data: station_no (PK), lat, lng, district.

Relationship: One-to-Many (1..*) with status logs.

station_status (Fact Table):

Stores time-series metrics: bikes_available, spaces_available, record_time.

High-frequency ingestion (every minute).

🛠️ Tech Stack
Cloud & DevOps: Google Cloud Platform (VM), Docker, Docker Compose, GCP Secret Manager.

Data Engineering: Apache Airflow, Python (Pandas), MySQL, SQLAlchemy.

Machine Learning: PyTorch (LSTM), Scikit-Learn (MinMaxScaler).

External APIs: YouBike 2.0 Open Data, Open-Meteo (Weather).

⚡ Key Features
🔐 1. Enterprise-Grade Security

Unlike typical student projects, this pipeline implements GCP Secret Manager to handle credentials.

Workflow: Airflow DAG → Request Secret (mysql_password) → GCP IAM Authentication → Return Payload → Connect to DB.

Benefit: Prevents credential leakage in version control (Git).

🧠 2. Multivariate LSTM Modeling

The prediction model goes beyond simple autoregression by incorporating environmental factors.

Input Features: [bikes_available, temperature, rain]

Model Architecture:

Layer 1: LSTM (Input: 3, Hidden: 64, Dropout: 0.2)

Layer 2: Fully Connected Layer

Result: Training Loss converged to 0.0130, successfully capturing traffic drops during rainfall.

🐳 3. Containerized Deployment

The entire ETL stack (Airflow Webserver, Scheduler, MySQL) is defined in docker-compose.yaml, ensuring Infrastructure as Code (IaC) and reproducibility across environments.

📊 Project Structure
Plaintext
YouBike-ETL-Pipeline/
├── dags/
│   └── youbike_dag.py     # Airflow DAG with GCP Secret Manager integration
├── data/
│   ├── raw/               # Raw CSV exports
│   └── processed/         # Merged dataset (Traffic + Weather)
├── docker-compose.yaml    # Services: Airflow, MySQL
├── Dockerfile             # Custom Image with GCP SDK installed
├── notebooks/
│   └── 04_lstm_prediction.ipynb  # PyTorch LSTM Training
└── requirements.txt       # Python dependencies
🚀 How to Run
Prerequisite

GCP Service Account with Secret Manager Secret Accessor role.

Docker & Docker Compose installed.

1. Deploy on GCP

Bash
# Clone the repository
git clone [Repo_URL]
cd YouBike-ETL-Pipeline

# Build and Start Services
docker-compose up -d --build
2. Configure Secrets

Ensure the secret mysql_password is created in GCP Secret Manager in the project youbike-airflow-server.

3. Run Analysis Locally

Export data from MySQL or use the provided CSVs in data/raw/.

Set up the Conda environment:

Bash
conda create -n youbike_ai python=3.10
conda activate youbike_ai
pip install -r requirements.txt
Execute notebooks/04_lstm_prediction.ipynb to train the model.

Created by [Kevin Lin] | 2025