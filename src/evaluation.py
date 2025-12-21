import os

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score
)

# =========================
# PATHS
# =========================
DATASET_PATH = "artifacts/factoryguard_features.joblib"
MODEL_PATH = "artifacts/final_production_model.joblib"
PLOT_PATH = "evaluation/plots/pr_curve.png"


# =========================
# LOAD DATA & MODEL
# =========================
def load_data_and_model():
    import pandas as pd

    # Load dataset
    df = joblib.load(DATASET_PATH)

    # Create synthetic binary failure label (for evaluation only)
    y = (df["vibration"] > 5.0).astype(int)


    # Use all features EXCEPT the target proxy
    X = df.drop(columns=["vibration"])

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    # Train-test split
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    return model, X_test, y_test


# =========================
# PR-AUC COMPUTATION
# =========================
def compute_pr_auc(model, X_test, y_test):
    """
    Compute PR-AUC using regression outputs as risk scores.
    """
    y_score = model.predict(X_test)
    pr_auc = average_precision_score(y_test, y_score)
    return pr_auc, y_score


# =========================
# PR CURVE PLOT
# =========================
def plot_pr_curve(y_test, y_score):
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()



# =========================
# THRESHOLD SELECTION
# =========================
def find_best_threshold(y_test, y_score):
    thresholds = np.arange(0.1, 0.91, 0.05)

    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        if precision > best_precision and recall > 0:
            best_precision = precision
            best_recall = recall
            best_threshold = t

    return best_threshold, best_precision, best_recall


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model, X_test, y_test = load_data_and_model()

    pr_auc, y_score = compute_pr_auc(model, X_test, y_test)
    plot_pr_curve(y_test, y_score)

    best_threshold, best_precision, best_recall = find_best_threshold(
        y_test, y_score
    )

    print("===== MODEL EVALUATION RESULTS =====")
    print(f"PR-AUC Score        : {pr_auc:.4f}")
    print(f"Best Threshold     : {best_threshold}")
    print(f"Precision @Thresh  : {best_precision:.4f}")
    print(f"Recall @Thresh     : {best_recall:.4f}")
    print(f"PR Curve Plot saved to: {PLOT_PATH}")
