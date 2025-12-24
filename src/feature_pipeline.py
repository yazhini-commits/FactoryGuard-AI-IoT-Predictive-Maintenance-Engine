import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------
# Base directory (src/)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / "data"
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# File paths
# -------------------------------------------------
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"
ROLLING_FEATURES_PATH = DATA_DIR / "rolling_features.csv"
LAG_FEATURES_PATH = DATA_DIR / "lag_feature_data.csv"

OUTPUT_FEATURES_PATH = ARTIFACTS_DIR / "factoryguard_features.joblib"


# -------------------------------------------------
# Load datasets
# -------------------------------------------------
def load_datasets():
    """Load cleaned, rolling, and lag feature datasets."""
    cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
    rolling_df = pd.read_csv(ROLLING_FEATURES_PATH)
    lag_df = pd.read_csv(LAG_FEATURES_PATH)

    print(" Datasets loaded successfully")
    return cleaned_df, rolling_df, lag_df


# -------------------------------------------------
# Merge features
# -------------------------------------------------
def merge_features(cleaned_df, rolling_df, lag_df):
    """Merge all features on machine_id and timestamp."""
    df = cleaned_df.merge(
        rolling_df,
        on=["machine_id", "timestamp"],
        how="inner"
    )

    df = df.merge(
        lag_df,
        on=["machine_id", "timestamp"],
        how="inner"
    )

    print(" Features merged successfully")
    return df


# -------------------------------------------------
# Save merged dataset
# -------------------------------------------------
def save_features(df):
    """Serialize merged features using joblib."""
    joblib.dump(df, OUTPUT_FEATURES_PATH)
    print(f" Features saved at: {OUTPUT_FEATURES_PATH}")


# -------------------------------------------------
# Load & display saved dataset
# -------------------------------------------------
def show_saved_features():
    """Load and display the saved merged dataset."""
    df = joblib.load(OUTPUT_FEATURES_PATH)

    print("\n Loaded merged dataset from joblib")
    print(f" Shape: {df.shape}")
    print("\n Columns:")
    print(df.columns.tolist())

    print("\n Sample rows:")
    print(df.head())


# -------------------------------------------------
# Run full pipeline
# -------------------------------------------------
def run_feature_pipeline():
    """Execute full feature integration pipeline."""
    print(" Starting feature engineering pipeline...\n")

    cleaned_df, rolling_df, lag_df = load_datasets()
    final_features = merge_features(cleaned_df, rolling_df, lag_df)

    save_features(final_features)
    show_saved_features()

    print("\n Feature engineering pipeline completed successfully!")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    run_feature_pipeline()
