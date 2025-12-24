import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path, sep=",")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['machine_id', 'timestamp'])

    df = df.groupby('machine_id').apply(
        lambda group: group.ffill()
    ).reset_index(drop=True)

    df.to_csv(output_path, index=False)

    print("Preprocessing completed successfully")
    print("Shape:", df.shape)
    print("Machines:", df['machine_id'].nunique())
    print("Time range:", df['timestamp'].min(), "to", df['timestamp'].max())


if __name__ == "__main__":
    input_path = BASE_DIR / "data" / "raw" / "sensor_data.csv"
    output_path = BASE_DIR / "data" / "processed" / "cleaned_data.csv"

    preprocess_data(input_path, output_path)
