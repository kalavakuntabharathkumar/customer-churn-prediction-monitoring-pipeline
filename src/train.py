import argparse, json
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.data import make_dataset, split_dataset, NUMERIC_FEATURES, CATEGORICAL_FEATURES

ARTIFACTS = Path("artifacts")

def build_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df = make_dataset(args.rows, args.seed)
    X_train, X_test, y_train, y_test = split_dataset(df, args.seed)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=8,
                                                 class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1)
    }
    try:
        from xgboost import XGBClassifier
        candidates["xgboost"] = XGBClassifier(n_estimators=220, max_depth=4, learning_rate=.05,
            subsample=.85, colsample_bytree=.85, eval_metric="logloss", random_state=args.seed,
            n_jobs=-1, tree_method="hist")
    except Exception as exc:
        print("XGBoost unavailable; continuing with sklearn models:", exc)
    best = None; best_score = -1; report = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:,1]
        # Choose by recall to align with churn-risk screening; inspect precision too.
        pred = (probs >= .40).astype(int)
        metrics = {
            "recall_at_0_40": float(recall_score(y_test, pred, zero_division=0)),
            "precision_at_0_40": float(precision_score(y_test, pred, zero_division=0)),
            "f1_at_0_40": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "accuracy_at_0_40": float(accuracy_score(y_test, pred)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        report[name] = metrics
        try:
            import mlflow
            with mlflow.start_run(run_name=name):
                mlflow.log_param("rows", args.rows)
                mlflow.log_param("threshold", .40)
                mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v, float)})
        except Exception:
            pass
        if metrics["recall_at_0_40"] > best_score:
            best_score = metrics["recall_at_0_40"]; best = (name, pipe, metrics)
        print(name, metrics)
    model_name, model, metrics = best
    joblib.dump(model, ARTIFACTS/"churn_model.joblib")
    metadata = {"model_name":model_name, "threshold":0.40, "features":{
        "tenure_months":"Customer tenure in months",
        "monthly_charges":"Monthly subscription charges",
        "support_tickets_90d":"Support tickets in the last 90 days",
        "usage_change_pct":"Usage change versus prior period, percent",
        "late_payments_6m":"Late payments in the last six months",
        "contract_type":"monthly, annual, or two_year",
        "autopay":"1 if autopay enabled, else 0",
        "product_count":"Number of subscribed products"},
        "training_rows":len(df), "seed":args.seed, "synthetic_data":True, "metrics":metrics,
        "candidate_metrics":report}
    (ARTIFACTS/"model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {model_name} model to artifacts/churn_model.joblib")

if __name__ == "__main__":
    main()
