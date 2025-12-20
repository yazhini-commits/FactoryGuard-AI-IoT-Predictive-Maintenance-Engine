import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score

from imblearn.over_sampling import SMOTE


DATA_PATH = "data/processed/final_features.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["failure"])
y = df["failure"]

print("\nClass Distribution:")
print(Counter(y))
print("Failure Rate (%):", round(y.mean() * 100, 4))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


cw_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

cw_model.fit(X_train, y_train)

y_prob_cw = cw_model.predict_proba(X_test)[:, 1]
y_pred_cw = (y_prob_cw >= 0.5).astype(int)

pr_auc_cw = average_precision_score(y_test, y_prob_cw)
precision_cw = precision_score(y_test, y_pred_cw, zero_division=0)

print("\nClass Weight Results")
print("PR-AUC     :", round(pr_auc_cw, 4))
print("Precision  :", round(precision_cw, 4))


smote = SMOTE(random_state=42)

X_sm, y_sm = smote.fit_resample(X_train, y_train)

smote_model = LogisticRegression(max_iter=1000)
smote_model.fit(X_sm, y_sm)

y_prob_sm = smote_model.predict_proba(X_test)[:, 1]
y_pred_sm = (y_prob_sm >= 0.5).astype(int)

pr_auc_sm = average_precision_score(y_test, y_prob_sm)
precision_sm = precision_score(y_test, y_pred_sm, zero_division=0)

print("\nSMOTE Results")
print("PR-AUC     :", round(pr_auc_sm, 4))
print("Precision  :", round(precision_sm, 4))


print("\n===== IMBALANCE STRATEGY COMPARISON =====")
print(f"Class Weights -> PR-AUC: {pr_auc_cw:.4f}, Precision: {precision_cw:.4f}")
print(f"SMOTE         -> PR-AUC: {pr_auc_sm:.4f}, Precision: {precision_sm:.4f}")

