import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "final_features.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
OUTPUT_DIR = BASE_DIR / "shap_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded data with shape: {df.shape}")

model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

X = df.drop(columns=["failure", "machine_id", "timestamp"])
y = df["failure"]

failure_cases = df[df["failure"] == 1].head(3)
print(f"Explaining {len(failure_cases)} failure cases")

X_fail = failure_cases.drop(
    columns=["failure", "machine_id", "timestamp"]
)

explainer = shap.Explainer(model, X)
shap_values = explainer(X_fail)

for i in range(len(X_fail)):
    plt.figure()
    shap.plots.waterfall(shap_values[i], show=False)

    file_path = OUTPUT_DIR / f"waterfall_failure_case_{i+1}.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {file_path.name}")

for i in range(len(X_fail)):
    plt.figure()
    shap.plots.force(
        explainer.expected_value,
        shap_values[i].values,
        X_fail.iloc[i],
        matplotlib=True,
        show=False
    )

    file_path = OUTPUT_DIR / f"force_failure_case_{i+1}.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {file_path.name}")
print("\nLocal SHAP explanations generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
