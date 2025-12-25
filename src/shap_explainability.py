# ==================================================
# Week 3 – Model Explainability using SHAP
# ==================================================

import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "final_production_model.joblib"
FEATURE_PATH = BASE_DIR / "artifacts" / "factoryguard_features.joblib"

OUTPUT_DIR = BASE_DIR  / "shap_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "failure"

# --------------------------------------------------
# Load trained model (Week 2 output)
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
print(" Trained model loaded")

# --------------------------------------------------
# Load feature dataset (Week 1 output)
# --------------------------------------------------
data = joblib.load(FEATURE_PATH)
data.columns = data.columns.str.lower()

# Drop non-feature columns
drop_cols = ['timestamp', 'machine_id', TARGET_COLUMN]
X = data.drop(columns=[c for c in drop_cols if c in data.columns])

# Convert all features to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Handle missing values
X = X.fillna(X.median())

print(f" Feature matrix prepared | Shape: {X.shape}")

# --------------------------------------------------
# SHAP Explainer Initialization
# --------------------------------------------------
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

print(" SHAP values computed")

# --------------------------------------------------
# 1. SHAP Summary Plot (Global Explainability)
# --------------------------------------------------
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig(
    OUTPUT_DIR / "shap_summary_plot.png",
    bbox_inches="tight",
    dpi=300
)
plt.close()

print(" SHAP summary plot saved")

# --------------------------------------------------
# 2. SHAP Bar Plot (Mean Absolute Impact)
# --------------------------------------------------
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.savefig(
    OUTPUT_DIR / "shap_bar_plot.png",
    bbox_inches="tight",
    dpi=300
)
plt.close()

print(" SHAP bar plot saved")

# --------------------------------------------------
# 3. Identify Top 5 Failure-Driving Features
# --------------------------------------------------
shap_importance = pd.DataFrame({
    "feature": X.columns,
    "mean_absolute_shap": np.abs(shap_values.values).mean(axis=0)
})

top_5_features = (
    shap_importance
    .sort_values(by="mean_absolute_shap", ascending=False)
    .head(5)
)

print("\n Top 5 Features Influencing Failure Prediction ")
print(top_5_features)

# Save top features
top_5_features.to_csv(
    OUTPUT_DIR / "top_5_failure_features.csv",
    index=False
)

print("\n Top 5 features saved to CSV")

# --------------------------------------------------
# END
# --------------------------------------------------
print("\nSHAP Explainability completed successfully!")
print(f"Outputs saved in: {OUTPUT_DIR}")
