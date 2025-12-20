import pandas as pd


def preprocess_data(input_path, output_path):
    # Load raw data
    df = pd.read_csv(input_path)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by machine and time
    df = df.sort_values(by=['machine_id', 'timestamp'])

    # Handle missing values
    df = df.groupby('machine_id').apply(
        lambda group: group.fillna(method='ffill')
    ).reset_index(drop=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)

    # Validation logs
    print("Preprocessing completed successfully")
    print("Shape:", df.shape)
    print("Machines:", df['machine_id'].nunique())
    print("Time range:", df['timestamp'].min(), "to", df['timestamp'].max())


if __name__ == "__main__":
    preprocess_data(
        input_path="data/raw/sensor_data.csv",
        output_path="data/processed/cleaned_data.csv"
    )
