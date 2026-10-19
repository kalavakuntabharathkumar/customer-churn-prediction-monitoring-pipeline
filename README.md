# Customer Churn Prediction & Monitoring Pipeline

A reproducible, end-to-end churn ML project with synthetic data generation, scikit-learn/XGBoost model comparison, SHAP-compatible explanations, MLflow experiment tracking, a FastAPI prediction service, Docker packaging, CI, and an Azure ML deployment template.

> **Data and impact note:** This repository uses synthetic customer data. Any model scores and financial-impact figures produced locally are demo outputs, not validated business outcomes. The reported 78% recall / ₹18 lakh annual savings in a resume bullet should only be claimed if independently reproduced on real, authorized data.

## Quick start (local)
Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train --rows 20000 --seed 42
uvicorn src.api:app --reload
```
Open `http://127.0.0.1:8000/docs` for API docs.

## Run with Docker
```bash
docker build -t churn-monitor .
docker run --rm -p 8000:8000 churn-monitor
```
Train the model first; generated artifacts are saved in `artifacts/`.

## Train on a larger dataset
```bash
python -m src.train --rows 380000 --seed 42
```
The larger setting takes longer and may require several GB of memory depending on hardware.

## API
- `GET /health` — readiness and artifact status
- `POST /predict` — churn probability, predicted label, and risk tier
- `GET /model-info` — model metadata
- `POST /monitor/batch` — calculate a simple batch drift report from JSON feature rows

Example:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 8, "monthly_charges": 79.5, "support_tickets_90d": 3,
  "usage_change_pct": -22, "late_payments_6m": 2, "contract_type": "monthly",
  "autopay": false, "product_count": 1
}'
```

## MLflow
Training logs metrics and parameters to local `mlruns/` when MLflow is installed. Start the UI with:
```bash
mlflow ui --backend-store-uri ./mlruns
```

## Azure ML
`azure/deploy_azureml.py` is a deployment scaffold that expects an existing Azure ML workspace and credentials. It is not automatically deployed by running the local pipeline. Review cloud costs and access policies before deployment.

## Structure
```text
src/                  data, training, API, monitoring
tests/                smoke and API tests
artifacts/             generated model and metadata
azure/                 deployment scaffold
.github/workflows/     CI workflow
```
