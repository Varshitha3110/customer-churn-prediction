import pandas as pd

from src.data_preprocessing import build_preprocessor, split_features_target


def test_split_features_target():
    df = pd.DataFrame(
        {
            "tenure": [1, 20, 40, 60],
            "MonthlyCharges": [30.0, 50.0, 80.0, 100.0],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Two year",
            ],
            "Churn": ["Yes", "No", "No", "Yes"],
        }
    )

    X, y = split_features_target(df)

    assert "Churn" not in X.columns
    assert list(y) == [1, 0, 0, 1]


def test_preprocessor_can_be_built():
    X = pd.DataFrame(
        {
            "tenure": [1, 20],
            "MonthlyCharges": [30.0, 50.0],
            "Contract": ["Month-to-month", "One year"],
        }
    )

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= 3
