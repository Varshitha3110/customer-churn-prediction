from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    build_preprocessor,
    load_data,
    split_features_target,
)
from src.model_training import get_model_configs, tune_model


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
VIZ_DIR = ROOT / "visualizations"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    VIZ_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Rows after cleaning: {len(df)}")

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results = []
    best_model = None
    best_name = None
    best_f1 = -1
    best_predictions = None
    best_probability = None

    for name, (model, param_grid) in get_model_configs().items():
        print(f"\nTraining: {name}")

        preprocessor = build_preprocessor(X_train)

        fitted_model, best_params = tune_model(
            preprocessor,
            model,
            param_grid,
            X_train,
            y_train,
            cv=3,
        )

        predictions = fitted_model.predict(X_test)

        probability = None
        if hasattr(fitted_model, "predict_proba"):
            probability = fitted_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = (
            roc_auc_score(y_test, probability)
            if probability is not None
            else float("nan")
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC_AUC": roc_auc,
                "Best_Params": best_params,
            }
        )

        print(
            f"Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"ROC-AUC={roc_auc:.4f}"
        )

        # F1 is used for model selection because churn is a business-sensitive class.
        if f1 > best_f1:
            best_f1 = f1
            best_model = fitted_model
            best_name = name
            best_predictions = predictions
            best_probability = probability

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    joblib.dump(best_model, MODEL_DIR / "churn_model.joblib")

    report = classification_report(
        y_test,
        best_predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )

    with open(
        REPORT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Best model: {best_name}\n")
        file.write("Selection metric: F1-score\n\n")
        file.write(report)

    cm = confusion_matrix(y_test, best_predictions)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["No Churn", "Churn"])
    ax.set_yticks([0, 1], ["No Churn", "Churn"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(
        VIZ_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = results_df.sort_values("F1")
    ax.barh(plot_df["Model"], plot_df["F1"])
    ax.set_xlabel("F1 Score")
    ax.set_title("Model Comparison by F1 Score")
    fig.tight_layout()
    fig.savefig(
        VIZ_DIR / "model_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    if best_probability is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(
            y_test,
            best_probability,
            ax=ax,
            name=best_name,
        )
        ax.set_title("ROC Curve - Best Model")
        fig.tight_layout()
        fig.savefig(
            VIZ_DIR / "roc_curve.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    print("\n" + "=" * 60)
    print(f"Best model: {best_name}")
    print(f"Best F1 score: {best_f1:.4f}")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nTraining failed: {exc}")
        sys.exit(1)
