# Taipei YouBike 2.0 AIoT Traffic Prediction System
### Cloud-Native Data Engineering | Multi-Station LSTM | Statistical Analysis

## Project Overview
This project is an **Enterprise-Grade End-to-End Data Engineering & Analytics solution** designed to address urban mobility challenges in Taipei City. By leveraging a **Cloud-Native architecture**, the system processes over **4,000,000+ real-time records** to optimize YouBike 2.0 station balancing.

It uniquely bridges the gap between **Modern Data Engineering** (GCP, Airflow, Docker) and **Rigorous Statistical Analysis** (Hypothesis Testing, ANOVA), aligning directly with UN SDGs.

### SDG Alignment
* **SDG 11 (Sustainable Cities):** Optimizing public transport availability to reduce private vehicle dependency.
* **SDG 13 (Climate Action):** Quantifying the impact of extreme weather events (e.g., heavy rainfall) on green transportation usage.

## Key Analytical Insights
We conducted comprehensive statistical testing (T-Test, ANOVA, Chi-Square) on the dataset. Detailed analysis reveals the following critical insights:

### 1. The "Average" Trap
* **Insight:** While Peak and Off-Peak hours share similar average usage (~36%), the **Coefficient of Variation (CV)** spikes to **0.78** during peak hours.
* **Conclusion:** The system faces extreme instability (high entropy) during rush hours, where "Stock-out" and "Full-load" events occur simultaneously. The problem is allocation, not total supply.

### 2. The Campus "Black Hole" Effect
* **Method:** Independent Samples T-Test & Forest Plot.
* **Finding:** The **NTU Gongguan Campus** area shows a persistent operational deficit compared to the nearby Commercial District (Da'an), with a **38.5% Stock-out Rate**.
* **Action:** A dedicated "Campus Shuttle" replenishment strategy is required, independent of district boundaries.

### 3. Land Use Zoning Effect
* **Method:** K-Means Clustering & One-way ANOVA.
* **Finding:** Operations follow a distinct hierarchy: **Mixed Use (Wanhua) > Commercial (Xinyi) > Residential (Wenshan)**.
* **Implication:** Mixed Zones suffer from high retention (need space strategy), while Commercial Zones suffer from high turnover (need speed strategy).

### 4. Model Evolution
* **M1 Static Model (R-squared = 0.02):** Location alone cannot predict bike availability.
* **M3 Dynamic Model (R-squared = 0.92):** Introducing "Lag Features" (High-frequency crawling data) proves that the system has strong state persistence. **Real-time data ingestion is business-critical.**

## System Architecture
The system is deployed on **Google Cloud Platform (GCP)** using a microservices pattern orchestrated by `docker-compose`.

### 1. Data Ingestion Layer (The ETL Pipeline)
* **Orchestration:** Apache Airflow runs scheduled DAGs (Crontab: Every 10 min) to trigger data ingestion.
* **Security:** **GCP Secret Manager** is integrated to securely retrieve database credentials (`mysql_password`) at runtime. **No sensitive keys are hardcoded.**
* **Resilience:** Implements retry logic and error handling for API connection timeouts.

### 2. Storage Layer (The Data Warehouse)
* **Database:** **MySQL 8.0** (Dockerized).
* **Schema:** Separates Dimension Tables (`station_info`) from Fact Tables (`station_status`) to support 4M+ rows.

### 3. Analytics & Serving Layer
* **Model Training:** **PyTorch LSTM** network trained on multi-station sequences.
* **API Service:** **FastAPI** serves prediction inference via REST endpoints.
* **Frontend:** **Streamlit** provides an interactive dashboard for real-time visualization.

## Database Schema Design (MySQL)
The database is designed with a normalized relational schema to optimize storage efficiency.

* **`station_info` (Dimension Table):**
    * Stores static data: `station_no` (PK), `lat`, `lng`, `district`.
    * Relationship: One-to-Many (`1..*`) with status logs.

* **`station_status` (Fact Table):**
    * Stores time-series metrics: `bikes_available`, `spaces_available`, `record_time`.
    * High-frequency ingestion (every 10 minutes).

## Tech Stack
* **Cloud & DevOps:** Google Cloud Platform (VM), Docker, Docker Compose, **GCP Secret Manager**.
* **Data Engineering:** Apache Airflow, Python (Pandas), MySQL, SQLAlchemy.
* **Machine Learning:** **PyTorch (LSTM)**, Scikit-Learn (MinMaxScaler).
* **Web Services:** FastAPI (Backend), Streamlit (Frontend).
* **BI Tools:** Tableau Public.

## Key Features

### 1. Enterprise-Grade Security
Unlike typical student projects, this pipeline implements **GCP Secret Manager** to handle credentials.
* **Workflow:** Airflow DAG -> Request Secret (`mysql_password`) -> GCP IAM Authentication -> Return Payload -> Connect to DB.
* *Benefit:* Prevents credential leakage in version control (Git).

### 2. Robust ETL Design
The DAG splits incoming JSON into two streams:
* **Station Info:** Only writes new stations (Static metadata).
* **Station Status:** Appends time-series log data every 10 minutes.

### 3. Full-Stack Data App
The project includes a user-facing application layer defined in `docker-compose.yaml`. The Dashboard container communicates with the API container via the internal Docker network, ensuring isolation and performance.

## Project Structure
```text
YouBike-ETL-Pipeline/
├── api/
│   ├── app/
│   │   ├── main.py            # FastAPI Entrypoint
│   │   └── model_files/       # LSTM Models (.pth) & Scalers
│   └── Dockerfile.app         # Unified App Container
├── dags/
│   └── youbike_dag.py         # Airflow DAG with GCP Secret Manager
├── dashboard/
│   └── app.py                 # Streamlit Visualization App
├── data/
│   ├── raw/                   # Raw CSV Data
│   └── processed/             # Cleaned Data for ML
├── docker-compose.yaml        # Microservices Definition
├── Dockerfile                 # Custom Airflow Image
├── notebooks/
│   ├── 01_youbike_analysis.ipynb
│   ├── 02_weather_etl.ipynb
│   └── 05_multistation_lstm.ipynb  # Main Deep Learning Training
└── requirements.txt           # Python dependencies
```
<br>

##  How to Run

### Prerequisite
1.  GCP Compute Engine (e2-medium or higher).
2.  GCP Service Account with `Secret Manager Secret Accessor` role.
3.  Docker & Docker Compose installed.

### 1. Deploy on GCP
```bash
# Clone the repository
git clone [Repo_URL]
cd YouBike-ETL-Pipeline

# Build and Start Services
docker-compose up -d --build

```
### 2. Configure Secrets
Ensure the secret mysql_password is created in GCP Secret Manager in the project youbike-airflow-server.

###  3. Access the Production Endpoints
The system services are accessible via the VM 

External IP:
Airflow UI: http://34.105.181.XX:8080

FastAPI Docs: http://34.105.181.XX:8000/docs

Dashboard: http://34.105.181.XX:8501

(Note: Replace .XX with the specific IP address)

### 4. Model Retraining
To update the LSTM model with the latest collected data:

1. Navigate to notebooks/.
Open 05_multistation_lstm.ipynb.

2. Run all cells to query MySQL, preprocess data, and retrain the PyTorch model.

3. The new weights will be saved to api/app/model_files/.

### 5. (conditional) Run Analysis Locally
Export data from MySQL or use the provided CSVs in data/raw/.
Set up the Conda environment:
```Bash
conda create -n youbike_ai python=3.10
conda activate youbike_ai
pip install -r requirements.txt
```
Execute `notebooks/05_multistation_lstm.ipynb` to train the model.



Created by [Kevin Lin] | 2025