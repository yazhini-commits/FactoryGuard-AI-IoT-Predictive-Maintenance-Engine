import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------
# Base directory (src/)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# File paths
# -------------------------------------------------
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_data.csv"
ROLLING_FEATURES_PATH = PROCESSED_DIR / "rolling_features.csv"
LAG_FEATURES_PATH = PROCESSED_DIR / "lag_feature_data.csv"

OUTPUT_FEATURES_PATH = ARTIFACTS_DIR / "factoryguard_features.joblib"


# -------------------------------------------------
# Load datasets
# -------------------------------------------------
def load_datasets():
    """Load cleaned, rolling, and lag feature datasets."""
    cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
    rolling_df = pd.read_csv(ROLLING_FEATURES_PATH)
    lag_df = pd.read_csv(LAG_FEATURES_PATH)

    print("Datasets loaded successfully\n")
    return cleaned_df, rolling_df, lag_df


# -------------------------------------------------
# Preprocess keys
# -------------------------------------------------
def preprocess_keys(*dfs):
    """Ensure consistent dtypes for merge keys."""
    for df in dfs:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["machine_id"] = df["machine_id"].astype(str)

    print("Merge keys standardized (machine_id, timestamp)\n")


# -------------------------------------------------
# Merge features
# -------------------------------------------------
def merge_features(cleaned_df, rolling_df, lag_df):
    """Merge all features using LEFT joins."""
    df = cleaned_df.merge(
        rolling_df,
        on=["machine_id", "timestamp"],
        how="left",
        suffixes=("", "_roll")
    )

    df = df.merge(
        lag_df,
        on=["machine_id", "timestamp"],
        how="left",
        suffixes=("", "_lag")
    )

    print("Features merged successfully")
    print(f"Rows after merge: {df.shape[0]}\n")
    return df


# -------------------------------------------------
# Display rows clearly
# -------------------------------------------------
def inspect_rows(df):
    """Display rows and NaN summary for inspection."""
    print("Merged Dataset Inspection")
    print("-" * 50)
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample rows (first 10):")
    print(df.head(10))

    print("\nMissing values summary (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15))
    print("\n")


# -------------------------------------------------
# Save merged dataset
# -------------------------------------------------
def save_features(df):
    """Serialize merged features using joblib."""
    joblib.dump(df, OUTPUT_FEATURES_PATH)
    print(f"Features saved at: {OUTPUT_FEATURES_PATH}\n")


# -------------------------------------------------
# Load & display saved dataset
# -------------------------------------------------
def show_saved_features():
    """Load and display the saved merged dataset."""
    df = joblib.load(OUTPUT_FEATURES_PATH)

    print("Reloaded Dataset from Joblib")
    print("-" * 50)
    print("Shape:", df.shape)
    print("\nSample rows:")
    print(df.head(10))
    print("\n")


# -------------------------------------------------
# Run full pipeline
# -------------------------------------------------
def run_feature_pipeline():
    """Execute full feature integration pipeline (inspection mode)."""
    print("Starting feature engineering pipeline...\n")

    cleaned_df, rolling_df, lag_df = load_datasets()
    preprocess_keys(cleaned_df, rolling_df, lag_df)

    merged_df = merge_features(cleaned_df, rolling_df, lag_df)

    #  IMPORTANT: View rows here
    inspect_rows(merged_df)

    save_features(merged_df)
    show_saved_features()

    print("Feature engineering pipeline completed successfully!")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    run_feature_pipeline()

