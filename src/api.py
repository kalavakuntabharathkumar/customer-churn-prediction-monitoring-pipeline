from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.data import FEATURES

app = FastAPI(title="Customer Churn Risk API", version="1.0.0",
              description="Demo churn scoring API using synthetic training data.")
MODEL_PATH = Path("artifacts/churn_model.joblib")
META_PATH = Path("artifacts/model_metadata.json")
model = None
metadata = {}

def load_artifacts():
    global model, metadata
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    support_tickets_90d: int = Field(ge=0, le=100)
    usage_change_pct: float = Field(ge=-100, le=500)
    late_payments_6m: int = Field(ge=0, le=100)
    contract_type: str = Field(pattern="^(monthly|annual|two_year)$")
    autopay: bool
    product_count: int = Field(ge=1, le=20)

@app.on_event("startup")
def startup():
    load_artifacts()

@app.get("/health")
def health():
    return {"status":"ok", "model_loaded":model is not None}

@app.get("/model-info")
def model_info():
    if not metadata:
        return {"status":"not_trained", "message":"Run python -m src.train first."}
    return {k:metadata.get(k) for k in ["model_name","training_rows","synthetic_data","metrics","threshold"]}

@app.post("/predict")
def predict(payload: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run python -m src.train first.")
    row = payload.model_dump()
    row["autopay"] = int(row["autopay"])
    X = pd.DataFrame([{k:row[k] for k in FEATURES}], columns=FEATURES)
    probability = float(model.predict_proba(X)[0,1])
    threshold = float(metadata.get("threshold", .4))
    label = int(probability >= threshold)
    return {"churn_probability":round(probability, 4), "predicted_churn":label,
            "risk_tier":"high" if probability >= .65 else "medium" if probability >= threshold else "low",
            "threshold":threshold, "disclaimer":"Synthetic demo model; do not use for real customer decisions."}

@app.post("/monitor/batch")
def monitor_batch(rows: list[CustomerFeatures]):
    if not rows:
        raise HTTPException(status_code=400, detail="Provide at least one feature row.")
    frame = pd.DataFrame([{**r.model_dump(), "autopay":int(r.autopay)} for r in rows])
    report = {}
    for col in ["tenure_months","monthly_charges","support_tickets_90d","usage_change_pct","late_payments_6m","product_count"]:
        vals = frame[col].astype(float)
        report[col] = {"batch_mean":round(float(vals.mean()), 4), "batch_std":round(float(vals.std(ddof=0)),4)}
    # This is a descriptive batch summary, not a calibrated statistical drift detector.
    return {"rows_received":len(rows), "feature_summary":report,
            "monitoring_note":"Compare this report with a versioned training baseline before declaring drift."}
