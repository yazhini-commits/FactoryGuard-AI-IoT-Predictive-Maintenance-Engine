from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# Configuration
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_PATH = BASE_DIR / "artifacts" / "factoryguard_features.joblib"
OUTPUT_MODEL_PATH = BASE_DIR / "artifacts" / "final_production_model.joblib"
OUTPUT_REPORT_PATH = BASE_DIR / "artifacts" / "final_model_report.joblib"
TARGET_COLUMN = "failure"

# ------------------------------
# Load features
# ------------------------------
features = joblib.load(FEATURE_PATH)
features.columns = features.columns.str.strip().str.lower()

drop_cols = ["timestamp", "machine_id"]
features = features.drop(columns=[c for c in drop_cols if c in features.columns])
features = features.dropna(subset=[TARGET_COLUMN])

X = features.drop(TARGET_COLUMN, axis=1)
y = pd.to_numeric(features[TARGET_COLUMN], errors="coerce")
mask = y.notna()
X = X[mask]
y = y[mask].astype(int)

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.fillna(X.median())

# ------------------------------
# Class imbalance handling
# ------------------------------
scale_pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
lgb_class_weight = {0: 1, 1: int((y == 0).sum() / max((y == 1).sum(), 1))}
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ------------------------------
# Cross-validation
# ------------------------------
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ------------------------------
# XGBoost Optuna objective
# ------------------------------
def xgb_objective(trial):
    params = {
        "objective": "binary:logistic",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "random_state": 42,
        "n_jobs": -1
    }

    scores = []
    for tr, va in kf.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X.iloc[tr],
            y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            eval_metric="logloss",
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict_proba(X.iloc[va])[:, 1]
        scores.append(average_precision_score(y.iloc[va], preds))

    return np.mean(scores)

# ------------------------------
# LightGBM Optuna objective
# ------------------------------
def lgb_objective(trial):
    max_depth = trial.suggest_int("max_depth", 4, 8)
    num_leaves = 2 ** max_depth - 1

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    scores = []
    for tr, va in kf.split(X, y):
        model = lgb.LGBMClassifier(**params, class_weight=lgb_class_weight)
        model.fit(
            X.iloc[tr],
            y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            callbacks=[lgb.early_stopping(50)]
        )
        preds = model.predict_proba(X.iloc[va])[:, 1]
        scores.append(average_precision_score(y.iloc[va], preds))

    return np.mean(scores)

# ------------------------------
# Run Optuna
# ------------------------------
xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=100)

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=100)

xgb_best_score = xgb_study.best_value
lgb_best_score = lgb_study.best_value

xgb_best_params = xgb_study.best_params
lgb_best_params = lgb_study.best_params

# ------------------------------
# Train BOTH best models
# ------------------------------
xgb_final = xgb.XGBClassifier(
    **xgb_best_params,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)
xgb_final.fit(X, y)

lgb_final = lgb.LGBMClassifier(
    **lgb_best_params,
    class_weight=lgb_class_weight,
    random_state=42,
    verbose=-1
)
lgb_final.fit(X, y)

# ------------------------------
# Separate classification reports
# ------------------------------
xgb_pred = xgb_final.predict(X)
lgb_pred = lgb_final.predict(X)

xgb_report_dict = classification_report(
    y, xgb_pred, target_names=["No Failure", "Failure"], output_dict=True
)
lgb_report_dict = classification_report(
    y, lgb_pred, target_names=["No Failure", "Failure"], output_dict=True
)

xgb_report_text = classification_report(
    y, xgb_pred, target_names=["No Failure", "Failure"]
)
lgb_report_text = classification_report(
    y, lgb_pred, target_names=["No Failure", "Failure"]
)

# ------------------------------
# Select final production model
# ------------------------------
if lgb_best_score > xgb_best_score:
    final_model = lgb_final
    final_model_name = "LightGBM"
    final_score = lgb_best_score
else:
    final_model = xgb_final
    final_model_name = "XGBoost"
    final_score = xgb_best_score

# ------------------------------
# Save artifacts
# ------------------------------
joblib.dump(final_model, OUTPUT_MODEL_PATH)

report = {
    "task": "binary_classification",
    "target_column": TARGET_COLUMN,
    "selected_model": final_model_name,
    "final_pr_auc": final_score,
    "xgboost": {
        "pr_auc": xgb_best_score,
        "best_params": xgb_best_params,
        "classification_report": xgb_report_dict
    },
    "lightgbm": {
        "pr_auc": lgb_best_score,
        "best_params": lgb_best_params,
        "classification_report": lgb_report_dict
    }
}

joblib.dump(report, OUTPUT_REPORT_PATH)

# ------------------------------
# Console output
# ------------------------------
print("\n===== XGBOOST CLASSIFICATION REPORT =====")
print(xgb_report_text)

print("\n===== LIGHTGBM CLASSIFICATION REPORT =====")
print(lgb_report_text)

print("\n===== FINAL MODEL SELECTION =====")
print(f"Selected Model        : {final_model_name}")
print(f"Cross-validated PR-AUC: {final_score:.6f}")
