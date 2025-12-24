from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
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

# Drop irrelevant columns
drop_cols = ['timestamp', 'machine_id']
features = features.drop(columns=[c for c in drop_cols if c in features.columns])

# Drop missing target rows
features = features.dropna(subset=[TARGET_COLUMN])

# Features and target
X = features.drop(TARGET_COLUMN, axis=1)
y = pd.to_numeric(features[TARGET_COLUMN], errors='coerce')
mask = y.notna()
X = X[mask]
y = y[mask].astype(int)

# Convert object columns to numeric
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Impute missing values with median
X = X.fillna(X.median())

# ------------------------------
# Compute scale_pos_weight for XGBoost
# ------------------------------
scale_pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ------------------------------
# Stratified K-Fold cross-validation
# ------------------------------
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ------------------------------
# XGBoost objective for Optuna
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
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_val.sum() == 0:
            continue

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='logloss',  # safe for rare positives
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, preds))
    return np.mean(scores) if scores else 0.0

# ------------------------------
# LightGBM objective for Optuna
# ------------------------------
def lgb_objective(trial):
    max_depth = trial.suggest_int("max_depth", 4, 8)
    num_leaves = 2 ** max_depth - 1  # avoid LightGBM leaf warnings

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
        "min_child_samples": 20,
        "min_split_gain": 0.0,
        "verbose": -1
    }

    # Compute class weights for imbalance
    weights = {0:1, 1:int((y==0).sum()/max((y==1).sum(),1))}
    
    scores = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_val.sum() == 0:
            continue

        model = lgb.LGBMClassifier(**params, class_weight=weights)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='average_precision',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        preds = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, preds))

    return np.mean(scores) if scores else 0.0

# ------------------------------
# Run Optuna studies
# ------------------------------
xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=100)
xgb_best_score = xgb_study.best_value
xgb_best_params = xgb_study.best_params

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=100)
lgb_best_score = lgb_study.best_value
lgb_best_params = lgb_study.best_params

# ------------------------------
# Select final model
# ------------------------------
if lgb_best_score > xgb_best_score:
    final_model = lgb.LGBMClassifier(
        **lgb_best_params,
        random_state=42,
        class_weight={0:1, 1:int((y==0).sum()/max((y==1).sum(),1))},
        verbose=-1
    )
    final_model_name = "LightGBM"
    final_score = lgb_best_score
else:
    final_model = xgb.XGBClassifier(
        **xgb_best_params,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )
    final_model_name = "XGBoost"
    final_score = xgb_best_score

# Fit final model on full dataset
final_model.fit(X, y)

# Save model and report
joblib.dump(final_model, OUTPUT_MODEL_PATH)

report = {
    "task": "binary_classification",
    "target_column": TARGET_COLUMN,
    "selected_model": final_model_name,
    "final_pr_auc": final_score,
    "xgboost_pr_auc": xgb_best_score,
    "xgboost_best_params": xgb_best_params,
    "lightgbm_pr_auc": lgb_best_score,
    "lightgbm_best_params": lgb_best_params
}

joblib.dump(report, OUTPUT_REPORT_PATH)

# ------------------------------
# Final report
# ------------------------------
print("\n===== FINAL PRODUCTION MODEL REPORT =====")
print(f"Selected Model       : {final_model_name}")
print(f"Cross-validated PR-AUC: {final_score:.6f}")
print(f"XGBoost PR-AUC       : {xgb_best_score:.6f}")
print(f"LightGBM PR-AUC      : {lgb_best_score:.6f}")
print("========================================")
