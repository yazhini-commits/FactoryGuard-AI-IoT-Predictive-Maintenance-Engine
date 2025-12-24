import pandas as pd
import os

# ------------------- Load CSV -------------------

input_file = "data/processed/cleaned_data.csv"  # updated path
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pd.read_csv(input_file)

# ------------------- Preprocessing -------------------

# Make column names lowercase
df.columns = df.columns.str.lower()

# Check required columns
required_cols = ['machine_id','timestamp','vibration','temperature','pressure','failure']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Parse timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['timestamp'])

# Ensure sensor columns are numeric
sensor_cols = ['vibration','temperature','pressure']
for sensor in sensor_cols:
    df[sensor] = pd.to_numeric(df[sensor], errors='coerce')

# Ensure failure is numeric
df['failure'] = pd.to_numeric(df['failure'], errors='coerce').fillna(0)

print(f"Data loaded successfully. Rows: {len(df)}")
print(df.head())

# ------------------- Rolling Features Function -------------------

def compute_time_based_rolling_features(df, sensors=['vibration','temperature','pressure'], windows=[1,6,12]):
    """
    Compute time-based rolling features per machine:
    - mean, std, EMA for sensors
    - rolling failure rate
    """
    df = df.copy()
    df = df.sort_values(['machine_id','timestamp'])
    results = []

    for machine_id, group in df.groupby('machine_id'):
        group = group.set_index('timestamp')

        for sensor in sensors:
            for w in windows:
                # Time-based rolling (hours)
                group[f'{sensor}_roll_mean_{w}h'] = group[sensor].rolling(f'{w}H', min_periods=1).mean()
                group[f'{sensor}_roll_std_{w}h'] = group[sensor].rolling(f'{w}H', min_periods=1).std().fillna(0)

            # EMA with default span 6
            group[f'{sensor}_ema'] = group[sensor].ewm(span=6, adjust=False).mean()

        # Rolling failure rate
        for w in windows:
            group[f'failure_roll_mean_{w}h'] = group['failure'].rolling(f'{w}H', min_periods=1).mean()

        results.append(group.reset_index())

    if not results:
        raise ValueError("No valid groups found for rolling computation.")

    return pd.concat(results, ignore_index=True)

# ------------------- Compute Rolling Features -------------------

window_hours = [1,6,12]
df_features = compute_time_based_rolling_features(df, sensors=sensor_cols, windows=window_hours)

# ------------------- Save CSV -------------------

output_file = "data/processed/rolling_features.csv"  # save in same folder
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_features.to_csv(output_file, index=False)
print(f"Rolling features created successfully!\nSaved to: {output_file}")

# Preview
print(df_features.head(10))
 
