import pandas as pd
from typing import List


def add_lag_features(
    df: pd.DataFrame,
    sensor_columns: List[str],
    lags: List[int],
    target_col: str = "failure"
) -> pd.DataFrame:
    """
    Create lag features (t-1, t-2, ...) grouped by machine_id
    and ensure target column is last.
    """

    df = df.copy()

    # Ensure correct ordering for time-series
    df = df.sort_values(
        by=["machine_id", "timestamp"]
    ).reset_index(drop=True)

    # Create lag features
    for sensor in sensor_columns:
        for lag in lags:
            df[f"{sensor}_t-{lag}"] = (
                df.groupby("machine_id")[sensor].shift(lag)
            )

    # Drop rows with NaNs introduced by lagging
    df = df.dropna().reset_index(drop=True)

    # ---- ENSURE FAILURE IS LAST COLUMN ----
    cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[cols]

    return df


if __name__ == "__main__":

    df = pd.read_csv("data/processed/cleaned_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    sensors = ["vibration", "temperature", "pressure"]
    lags = [1, 2]

    df_lagged = add_lag_features(
        df=df,
        sensor_columns=sensors,
        lags=lags,
        target_col="failure"
    )

    df_lagged.to_csv(
        "data/processed/lag_feature_data.csv",
        index=False
    )

    print("Lag features created successfully")
    print("Columns order:")
    print(df_lagged.columns.tolist())
    print("\nSample data:")
    print(df_lagged.head())
