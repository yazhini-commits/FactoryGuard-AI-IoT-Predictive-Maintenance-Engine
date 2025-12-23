import os
import joblib
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

# ------------------------------
# 0. Configuration
# ------------------------------
FEATURE_PATH = "../artifacts/factoryguard_features.joblib"
OUTPUT_MODEL_PATH = "../artifacts/final_production_model.joblib"
OUTPUT_REPORT_PATH = "../artifacts/final_model_report.joblib"
TARGET_COLUMN = "vibration" 

# Suppress warnings globally
warnings.filterwarnings("ignore")

# ------------------------------
# 1. Load features
# ------------------------------
features = joblib.load(FEATURE_PATH)
print("✅ Features loaded")
print("Columns:", features.columns)

if TARGET_COLUMN not in features.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found!")

# ------------------------------
# 2. Prepare X and y
# ------------------------------
X = features.drop(TARGET_COLUMN, axis=1)
y = features[TARGET_COLUMN].copy()

# Convert object columns to numeric
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill NaNs with 0
X = X.fillna(0)
print("✅ Features converted to numeric, shape:", X.shape)

# ------------------------------
# 3. Train/validation split
# ------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 4. XGBoost + Optuna tuning
# ------------------------------
def xgb_objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "random_state": 42,
        "n_jobs": -1
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return -np.sqrt(mean_squared_error(y_val, preds)) 

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50)
xgb_best_score = xgb_study.best_value
xgb_best_params = xgb_study.best_params
print("✅ XGBoost best score (neg RMSE):", xgb_best_score)
print("✅ XGBoost best params:", xgb_best_params)

# ------------------------------
# 5. LightGBM + Optuna tuning
# ------------------------------
def lgb_objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "min_child_samples": 1,
        "min_split_gain": 0.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return -np.sqrt(mean_squared_error(y_val, preds))

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=50)
lgb_best_score = lgb_study.best_value
lgb_best_params = lgb_study.best_params
print("✅ LightGBM best score (neg RMSE):", lgb_best_score)
print("✅ LightGBM best params:", lgb_best_params)

# ------------------------------
# 6. Select best model
# ------------------------------
if lgb_best_score > xgb_best_score:
    final_model = lgb.LGBMRegressor(**lgb_best_params, random_state=42,
                                    min_child_samples=1, min_split_gain=0.0, verbose=-1)
    final_model_name = "LightGBM"
    final_score = lgb_best_score
else:
    final_model = xgb.XGBRegressor(**xgb_best_params, random_state=42)
    final_model_name = "XGBoost"
    final_score = xgb_best_score

# Retrain final model on the full training set
final_model.fit(X_train, y_train)

# ------------------------------
# 7. Save model and report
# ------------------------------
joblib.dump(final_model, OUTPUT_MODEL_PATH)

report = {
    "selected_model": final_model_name,
    "final_score_neg_rmse": final_score,
    "xgboost_best_score": xgb_best_score,
    "xgboost_best_params": xgb_best_params,
    "lightgbm_best_score": lgb_best_score,
    "lightgbm_best_params": lgb_best_params
}

joblib.dump(report, OUTPUT_REPORT_PATH)

# ------------------------------
# 8. Print final report
# ------------------------------
print("\n===== FINAL PRODUCTION MODEL REPORT =====")
print(f"Selected Model       : {final_model_name}")
print(f"Final Score (neg RMSE): {final_score:.6f}")
print(f"XGBoost Score        : {xgb_best_score:.6f}")
print(f"LightGBM Score       : {lgb_best_score:.6f}")
print("========================================")
