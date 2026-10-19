"""Optional SHAP helper. Usage: python -m src.explain (requires trained artifacts and SHAP)."""
from pathlib import Path
import joblib
from src.data import make_dataset

def main():
    model_path = Path("artifacts/churn_model.joblib")
    if not model_path.exists():
        raise SystemExit("Train first: python -m src.train")
    try:
        import shap
    except ImportError:
        raise SystemExit("Install SHAP with `pip install shap` to run explanations.")
    model = joblib.load(model_path)
    X = make_dataset(300, 99).drop(columns="churn")
    transformed = model.named_steps["preprocessor"].transform(X)
    explainer = shap.Explainer(model.named_steps["model"], transformed)
    values = explainer(transformed[:50])
    print("Computed SHAP values for 50 synthetic rows.")
    print("SHAP array shape:", values.values.shape)
    print("For production, preserve transformed feature names and validate explanation semantics.")

if __name__ == "__main__":
    main()
