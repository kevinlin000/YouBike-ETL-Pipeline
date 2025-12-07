# Taipei YouBike ETL Pipeline & Analytics

## Project Overview
This project is an end-to-end data engineering solution designed to analyze the usage patterns of YouBike 2.0 in Taipei City.
It aims to provide data-driven insights for sustainable urban planning (SDG 11).

## Tech Stack
- **Language:** Python 3.9+
- **Database:** MySQL 8.0
- **Infrastructure:** Docker (Planned), GCP (Planned)
- **Visualization:** Tableau

## Architecture
1. **Extract:** Python script fetches real-time data from YouBike API every 10 mins.
2. **Transform:** Data cleaning and normalization using Pandas.
3. **Load:** Storing structured data into MySQL (Station Info & Status Logs).