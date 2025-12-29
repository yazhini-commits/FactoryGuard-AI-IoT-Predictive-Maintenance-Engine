import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "final_production_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "factoryguard_features.joblib"

# -----------------------------
# Load Model
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Get expected feature names
if hasattr(model, "get_booster"):  # XGBoost
    FEATURE_COLUMNS = model.get_booster().feature_names
else:  # sklearn fallback
    FEATURE_COLUMNS = model.feature_names_in_

# -----------------------------
# Load Feature Dataset
# -----------------------------
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}")
df = joblib.load(FEATURES_PATH)

TARGET_COLS = ["failure", "failure_x", "failure_y"]
df = df.drop(columns=[c for c in TARGET_COLS if c in df.columns], errors="ignore")

# -----------------------------
# Prediction Function (REAL-TIME READY)
# -----------------------------
def predict(sensor_input: dict, threshold: float = 0.5):
    """
    Performs inference on a single sensor snapshot.

    Args:
        sensor_input (dict): JSON-like dict of sensor readings
        threshold (float): Probability threshold to flag failure

    Returns:
        dict: {
            "failure_probability": float,
            "failure_alert": bool
        }
    """
    # Convert JSON/dict to DataFrame row
    df_input = pd.DataFrame([sensor_input])

    # Add missing features
    for col in FEATURE_COLUMNS:
        if col not in df_input.columns:
            df_input[col] = 0

    # Ensure correct feature order
    df_input = df_input[FEATURE_COLUMNS]

    # Predict probability
    failure_prob = model.predict_proba(df_input)[0][1]

    return {
        "failure_probability": round(float(failure_prob), 6),
        "failure_alert": bool(failure_prob >= threshold)
    }

# -----------------------------
# Test Run (optional)
# -----------------------------
if __name__ == "__main__":
    sample_row = df.iloc[0].to_dict()  # convert to dict for API-like input
    result = predict(sample_row)
    print(result)
