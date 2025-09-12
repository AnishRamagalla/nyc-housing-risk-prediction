import sys
import pandas as pd
import joblib
import os

MODEL_PATH = "models/monthly_rf_v1.joblib"
FEATURES_FILE = "data/live/features.csv"

def align_columns(df, model):
    """Ensure dataframe columns match model training features"""
    need = list(model.feature_names_in_)
    for c in need:
        if c not in df.columns:
            df[c] = 0.0
    extra = [c for c in df.columns if c not in need]
    if extra:
        df = df.drop(columns=extra)
    return df[need]

def predict(building_id):
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"{FEATURES_FILE} not found. Run src/features.py first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train your model first.")

    df = pd.read_csv(FEATURES_FILE, parse_dates=["snapshot_date"])
    if "building_id" not in df.columns:
        raise ValueError("features.csv missing building_id column")

    row = df[df["building_id"].astype(str) == str(building_id)]
    if row.empty:
        raise ValueError(f"No features found for building_id {building_id}")

    model = joblib.load(MODEL_PATH)
    X = row.drop(columns=["building_id", "snapshot_date"], errors="ignore")
    X = align_columns(X, model)
    prob = model.predict_proba(X)[0,1]
    pred = int(prob >= 0.5)

    print(f"Building ID: {building_id}")
    print(f"Snapshot: {row['snapshot_date'].iloc[0]}")
    print(f"Predicted label: {pred}")
    print(f"Risk probability: {prob:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict_live.py <building_id>")
        sys.exit(1)
    bid = sys.argv[1]
    predict(bid)
