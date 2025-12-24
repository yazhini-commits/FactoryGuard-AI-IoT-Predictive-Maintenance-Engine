import pandas as pd
import os

def add_rolling_features(df, window_hours=[1, 6, 12]):
    """
    Create rolling window features for sensor data.

    Parameters:
    df : pandas.DataFrame
        Input dataframe containing:
        - timestamp
        - vibration
        - temperature
        - pressure
    window_hours : list
        Rolling window sizes in hours

    Returns:
    pandas.DataFrame
    """

    df = df.copy()

    # Convert timestamp to datetime and set index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()

    sensors = ["vibration", "temperature", "pressure"]

    for sensor in sensors:
        for w in window_hours:
            # Rolling Mean
            df[f"{sensor}_rolling_mean_{w}h"] = (
                df[sensor].rolling(window=f"{w}H", min_periods=1).mean()
            )

            # Rolling Standard Deviation
            df[f"{sensor}_rolling_std_{w}h"] = (
                df[sensor].rolling(window=f"{w}H", min_periods=1).std()
            )

        # Exponential Moving Average (EMA)
        df[f"{sensor}_ema"] = df[sensor].ewm(
            span=window_hours[1], adjust=False
        ).mean()

    return df


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":

    input_path = "data/processed/cleaned_data.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    df_features = add_rolling_features(df)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/sensor_rolling_features.csv"
    df_features.to_csv(output_path)

    print("Rolling window features created successfully.")
    print(f"Input file : {input_path}")
    print(f"Output file: {output_path}")
