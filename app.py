import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "final_production_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "factoryguard_features.joblib"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(MODEL_PATH)
EXPECTED_FEATURES = int(model.n_features_in_)

# -----------------------------
# LOAD FEATURES
# -----------------------------
feature_info = joblib.load(FEATURES_PATH)

if hasattr(feature_info, "columns"):
    FEATURE_COLUMNS = list(feature_info.columns)
else:
    FEATURE_COLUMNS = list(feature_info)

# FORCE FEATURE COUNT MATCH
FEATURE_COLUMNS = FEATURE_COLUMNS[:EXPECTED_FEATURES]

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_failure(form_data):

    df = pd.DataFrame([form_data])

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_COLUMNS]
    X = df.astype(float).to_numpy()

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
    else:
        prob = float(model.predict(X)[0])

    prediction = "Failure" if prob >= 0.5 else "No Failure"
    return round(prob, 4), prediction

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    prob, result = predict_failure(request.form)
    return render_template(
        "result.html",
        prediction=result,
        probability=prob
    )

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
