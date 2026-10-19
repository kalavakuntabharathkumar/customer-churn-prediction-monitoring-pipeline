import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["tenure_months","monthly_charges","support_tickets_90d","usage_change_pct",
            "late_payments_6m","contract_type","autopay","product_count"]
NUMERIC_FEATURES = ["tenure_months","monthly_charges","support_tickets_90d","usage_change_pct",
                    "late_payments_6m","autopay","product_count"]
CATEGORICAL_FEATURES = ["contract_type"]

def make_dataset(rows=20000, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    charges = np.clip(rng.normal(68, 22, rows), 15, 160)
    tickets = rng.poisson(1.2, rows).clip(0, 12)
    usage_change = np.clip(rng.normal(-2, 24, rows), -90, 100)
    late_payments = rng.poisson(.65, rows).clip(0, 8)
    contract = rng.choice(["monthly","annual","two_year"], rows, p=[.52,.30,.18])
    autopay = rng.binomial(1, .62, rows)
    products = rng.choice([1,2,3,4], rows, p=[.48,.32,.15,.05])
    logit = (-2.0 + .025*(charges-60) - .035*tenure + .23*tickets
             - .018*usage_change + .34*late_payments
             + .85*(contract=="monthly") + .5*(autopay==0) - .32*(products-1))
    probability = 1/(1+np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({
        "tenure_months":tenure, "monthly_charges":charges.round(2),
        "support_tickets_90d":tickets, "usage_change_pct":usage_change.round(2),
        "late_payments_6m":late_payments, "contract_type":contract,
        "autopay":autopay.astype(int), "product_count":products, "churn":churn
    })

def split_dataset(df, seed=42):
    X = df[FEATURES]
    y = df["churn"]
    return train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
