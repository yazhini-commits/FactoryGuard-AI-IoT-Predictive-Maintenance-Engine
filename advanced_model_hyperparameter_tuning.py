from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings

# ------------------------------
# 0. Configuration
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_PATH = BASE_DIR / "artifacts" / "factoryguard_features.joblib"
OUTPUT_MODEL_PATH = BASE_DIR / "artifacts" / "final_production_model.joblib"
OUTPUT_REPORT_PATH = BASE_DIR / "artifacts" / "final_model_report.joblib"

TARGET_COLUMN = "failure"

warnings.filterwarnings("ignore")

# ------------------------------
# 1. Load features
# ------------------------------
if not FEATURE_PATH.exists():
    raise FileNotFoundError(f" Feature file not found: {FEATURE_PATH}")

features = joblib.load(FEATURE_PATH)

# Normalize column names
features.columns = features.columns.str.strip().str.lower()

print(" Features loaded")
print("Columns:", list(features.columns))

if TARGET_COLUMN not in features.columns:
    raise ValueError(
        f" Target column '{TARGET_COLUMN}' not found.\n"
        f"Available columns: {list(features.columns)}"
    )

# ------------------------------
# 2. Prepare X and y
# ------------------------------
X = features.drop(TARGET_COLUMN, axis=1)
y = features[TARGET_COLUMN].astype(int)

# Convert non-numeric columns
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.fillna(0)

print(" Feature matrix shape:", X.shape)
print(" Target distribution:")
print(y.value_counts())

# ------------------------------
# 3. Train/validation split
# ------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------
# 4. XGBoost + Optuna (CLASSIFICATION)
# ------------------------------
def xgb_objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": 42,
        "n_jobs": -1
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50)

xgb_best_score = xgb_study.best_value
xgb_best_params = xgb_study.best_params

print(" XGBoost AUC:", xgb_best_score)

# ------------------------------
# 5. LightGBM + Optuna (CLASSIFICATION)
# ------------------------------
def lgb_objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=50)

lgb_best_score = lgb_study.best_value
lgb_best_params = lgb_study.best_params

print(" LightGBM AUC:", lgb_best_score)

# ------------------------------
# 6. Select best model
# ------------------------------
if lgb_best_score > xgb_best_score:
    final_model = lgb.LGBMClassifier(
        **lgb_best_params,
        random_state=42,
        verbose=-1
    )
    final_model_name = "LightGBM"
    final_score = lgb_best_score
else:
    final_model = xgb.XGBClassifier(
        **xgb_best_params,
        random_state=42,
        n_jobs=-1
    )
    final_model_name = "XGBoost"
    final_score = xgb_best_score

final_model.fit(X_train, y_train)

# ------------------------------
# 7. Save model & report
# ------------------------------
joblib.dump(final_model, OUTPUT_MODEL_PATH)

report = {
    "task": "binary_classification",
    "target_column": TARGET_COLUMN,
    "selected_model": final_model_name,
    "final_auc": final_score,
    "xgboost_auc": xgb_best_score,
    "xgboost_best_params": xgb_best_params,
    "lightgbm_auc": lgb_best_score,
    "lightgbm_best_params": lgb_best_params
}

joblib.dump(report, OUTPUT_REPORT_PATH)

# ------------------------------
# 8. Final report
# ------------------------------
print("\n===== FINAL PRODUCTION MODEL REPORT =====")
print(f"Task               : Binary Classification")
print(f"Target Column      : {TARGET_COLUMN}")
print(f"Selected Model     : {final_model_name}")
print(f"Final AUC Score    : {final_score:.6f}")
print("========================================")
