from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Churn"
ID_COLUMN = "customerID"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and clean the Telco Customer Churn dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Place the Telco CSV inside the data directory."
        )

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' was not found.")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    df = df.dropna().copy()

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    return df


def split_features_target(df: pd.DataFrame):
    """Separate features and binary churn target."""
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Churn contains values other than 'Yes' and 'No'.")

    return X, y.astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a reusable preprocessing transformer."""
    numeric = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
