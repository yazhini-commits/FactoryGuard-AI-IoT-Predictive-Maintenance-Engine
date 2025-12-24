import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "Artifacts"

DATA_PATH = ARTIFACTS_DIR / "factoryguard_features.joblib"

# -------------------------------------------------
# Step 1: Load dataset
# -------------------------------------------------
print("Loading merged feature dataset...")
df = joblib.load(DATA_PATH)

print("\nColumns in dataset:")
print(df.columns.tolist())

# -------------------------------------------------
# Step 2: Define target and features
# -------------------------------------------------
TARGET_COL = "failure"   #  FIXED (lowercase)

COLUMNS_TO_DROP = [
    TARGET_COL,
    "timestamp",
    "machine_id",
    "vibration_x", "temperature_x", "pressure_x",
    "vibration_y", "temperature_y", "pressure_y",
    "vibration", "temperature", "pressure"
]

X = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
y = df[TARGET_COL]

# Ensure numeric features
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.fillna(0)

# Sanity check
assert y.nunique() == 2, "Target must be binary (0/1)"

print("\nTarget distribution:")
print(y.value_counts())

# -------------------------------------------------
# Step 3: Stratified train-test split
# -------------------------------------------------
print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

print(f"Test size: {len(y_test)} | Failures in test: {y_test.sum()}")

# -------------------------------------------------
# Step 4: Logistic Regression
# -------------------------------------------------
print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
    random_state=42
)

lr.fit(X_train, y_train)

lr_probs = lr.predict_proba(X_test)[:, 1]
lr_pr_auc = average_precision_score(y_test, lr_probs)

print(f"Logistic Regression PR-AUC: {lr_pr_auc:.4f}")

# -------------------------------------------------
# Step 5: Random Forest
# -------------------------------------------------
print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_pr_auc = average_precision_score(y_test, rf_probs)

print(f"Random Forest PR-AUC: {rf_pr_auc:.4f}")

# -------------------------------------------------
# Step 6: Results table
# -------------------------------------------------
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "PR-AUC": [lr_pr_auc, rf_pr_auc]
})

print("\nBaseline Model Comparison:")
print(results)

# -------------------------------------------------
# Step 7: Save models
# -------------------------------------------------
joblib.dump(lr, ARTIFACTS_DIR / "baseline_logistic_regression.joblib")
joblib.dump(rf, ARTIFACTS_DIR / "baseline_random_forest.joblib")

print("\nBaseline models saved successfully")


