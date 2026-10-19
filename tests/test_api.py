from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_prediction_validation():
    response = client.post("/predict", json={
        "tenure_months": 8, "monthly_charges": 79.5, "support_tickets_90d": 3,
        "usage_change_pct": -22, "late_payments_6m": 2, "contract_type": "monthly",
        "autopay": False, "product_count": 1
    })
    assert response.status_code in (200, 503)
