"""
Lag Features Module for FactoryGuard AI

This module provides functions to create lag features from time-series sensor data,
which are essential for predictive maintenance models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_lag_features(df: pd.DataFrame,
                     columns: List[str],
                     lags: List[int],
                     suffix: str = '_lag') -> pd.DataFrame:
    """
    Add lag features to specified columns in the DataFrame.

    Args:
        df: Input DataFrame with time-series data
        columns: List of column names to create lags for
        lags: List of lag periods (e.g., [1, 2, 3] for 1-step, 2-step, 3-step lags)
        suffix: Suffix to append to new column names

    Returns:
        DataFrame with added lag features
    """
    df_lagged = df.copy()

    for col in columns:
        for lag in lags:
            lag_col_name = f"{col}{suffix}_{lag}"
            df_lagged[lag_col_name] = df_lagged.groupby("machine_id")[col].shift(lag)

    return df_lagged


def add_lag_features(df, sensors, lags):
    for sensor in sensors:
        for lag in lags:
            df[f"{sensor}_t-{lag}"] = (
                df.groupby("machine_id")[sensor].shift(lag)
            )
    return df


def add_rolling_features(df: pd.DataFrame,
                        columns: List[str],
                        windows: List[int],
                        operations: List[str] = ['mean', 'std'],
                        suffix: str = '_roll') -> pd.DataFrame:
    """
    Add rolling window features to specified columns.

    Args:
        df: Input DataFrame with time-series data
        columns: List of column names to create rolling features for
        windows: List of window sizes
        operations: List of operations to apply (mean, std, min, max, etc.)
        suffix: Suffix to append to new column names

    Returns:
        DataFrame with added rolling features
    """
    df_rolled = df.copy()

    for col in columns:
        for window in windows:
            for op in operations:
                roll_col_name = f"{col}{suffix}_{op}_{window}"
                df_rolled[roll_col_name] = df_rolled[col].rolling(window=window).agg(op)

    return df_rolled


def create_time_series_features(df: pd.DataFrame,
                               time_column: str,
                               sensor_columns: List[str],
                               lags: List[int] = [1, 2, 3],
                               windows: List[int] = [5, 10]) -> pd.DataFrame:
    """
    Create comprehensive time-series features for predictive maintenance.

    Args:
        df: Input DataFrame
        time_column: Name of the timestamp column
        sensor_columns: List of sensor data columns
        lags: Lag periods to create
        windows: Rolling window sizes

    Returns:
        DataFrame with lag and rolling features
    """
    # Ensure DataFrame is sorted by time
    df = df.sort_values(time_column).reset_index(drop=True)

    # Add lag features
    df = add_lag_features(df, sensor_columns, lags)

    # Add rolling features
    df = add_rolling_features(df, sensor_columns, windows)

    # Add time-based features
    df['hour'] = pd.to_datetime(df[time_column]).dt.hour
    df['day_of_week'] = pd.to_datetime(df[time_column]).dt.dayofweek

    return df


if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv("data/processed/cleaned_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort data properly for lag features
    df = df.sort_values(by=["machine_id", "timestamp"])

    # Create features for specified columns and lags
    sensors = ["vibration", "temperature", "pressure"]
    lags = [1, 2]

    df_lagged = add_lag_features(df, sensors, lags)

    # Save output for testing
    df_lagged.to_csv("data/processed/lag_feature_data.csv", index=False)
    print("Lag features created!")

    print("Data loaded and sorted:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("\nLag features created:")
    print(df_lagged.head(10))