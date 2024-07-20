import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "../models/heat_wave_model.pkl")
    print(f"Model saved to ../models/heat_wave_model.pkl")

if __name__ == "__main__":
    data = load_data("../data/processed_heat_wave_data.csv")
    train_model(data)
