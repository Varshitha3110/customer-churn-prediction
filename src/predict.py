from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path="models/churn_model.joblib"):
    """Load the trained model pipeline."""
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python train.py` first."
        )

    return joblib.load(path)


def predict_customer(customer: dict, model=None):
    """Predict churn for one customer."""
    if model is None:
        model = load_model()

    X = pd.DataFrame([customer])
    prediction = int(model.predict(X)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0][1])

    return {
        "churn": prediction,
        "label": "Yes" if prediction == 1 else "No",
        "probability": probability,
    }
