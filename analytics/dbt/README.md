# dbt Analytics Layer

This folder contains a lightweight dbt scaffold for documenting and testing the analytical data model built from the YouBike warehouse tables.

It is intentionally separate from the production Airflow ETL path:

- Airflow owns ingestion into MySQL.
- dbt owns downstream analytical transformations and data tests.

The project includes small dbt seeds so the analytics layer can be tested in CI against a disposable MySQL service. Use `profiles.example.yml` as a template for local or deployed MySQL credentials.

## Intended Layers

- `sources`: raw warehouse tables created by `sql/init_schema.sql`
- `seeds`: small CI fixtures that mimic the raw warehouse tables
- `staging`: cleaned column names and derived fields
- `marts`: analysis-ready tables for station health and hourly operations

## Example Commands

```bash
cd analytics/dbt
dbt deps
dbt seed --profiles-dir .
dbt parse --profiles-dir .
dbt build --profiles-dir .
```

Use `profiles.example.yml` to create a real `profiles.yml`. Do not commit credentials.
