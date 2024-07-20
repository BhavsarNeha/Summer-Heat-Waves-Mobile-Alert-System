import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Add your data preprocessing steps here
    return df

if __name__ == "__main__":
    data = load_data("../data/heat_wave_data.csv")
    processed_data = preprocess_data(data)
    processed_data.to_csv("../data/processed_heat_wave_data.csv", index=False)
