import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Paths to Artifacts
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "final_production_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "factoryguard_features.joblib"

# -----------------------------
# Load Model & Features
# -----------------------------
if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    raise FileNotFoundError("Model or feature file not found in artifacts.")

model = joblib.load(MODEL_PATH)
feature_info = joblib.load(FEATURES_PATH)

# Extract feature names
if hasattr(feature_info, "feature_names_in_"):
    FEATURE_COLUMNS = feature_info.feature_names_in_
elif hasattr(feature_info, "columns"):
    FEATURE_COLUMNS = list(feature_info.columns)
else:
    FEATURE_COLUMNS = list(feature_info)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_failure(sensor_input: dict, threshold: float = 0.5):
    """
    Perform real-time failure prediction
    """
    df_input = pd.DataFrame([sensor_input])

    # Fill missing features
    for col in FEATURE_COLUMNS:
        if col not in df_input.columns:
            df_input[col] = 0

    # Align feature order
    df_input = df_input[FEATURE_COLUMNS]

    # Apply preprocessing if available
    try:
        df_processed = feature_info.transform(df_input)
    except:
        df_processed = df_input

    # Predict probability
    if hasattr(model, "predict_proba"):
        failure_prob = model.predict_proba(df_processed)[0][1]
    else:
        failure_prob = float(model.predict(df_processed)[0])

    # Binary prediction
    failure_flag = failure_prob >= threshold

    return round(float(failure_prob), 6), "Failure" if failure_flag else "No Failure"

# -----------------------------
# POST /predict Endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    sensor_data = request.get_json()
    try:
        failure_prob, prediction = predict_failure(sensor_data)
        return jsonify({
            "failure_probability": failure_prob,
            "prediction": prediction
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# GET /predict Test-Friendly Endpoint
# -----------------------------
@app.route("/predict", methods=["GET"])
def predict_get_test():
    """
    Simple GET for browser check.
    """
    return jsonify({
        "message": "POST JSON to this endpoint to get failure prediction",
        "example": {
            "rolling_mean_temperature": 78.5,
            "rolling_std_vibration": 0.12,
            "pressure_trend": 101.3
        }
    }), 200

# -----------------------------
# Root Endpoint
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "IoT Predictive Maintenance API is running. Use POST /predict"}), 200

# -----------------------------
# Run Flask Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
