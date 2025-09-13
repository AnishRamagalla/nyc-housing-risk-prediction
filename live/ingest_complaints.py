# ingest_complaints.py

import pandas as pd
import os

RAW_DB = "data/raw/complaints.db"
LIVE_FEATURES = "data/live/features.csv"

def build_live_features():
    # Example: load processed building_features.csv (adapt to your pipeline)
    df = pd.read_csv("data/processed/building_features.csv")

    # Feature engineering
    df["events_3m"] = df["events_prev_year"] / 4
    df["events_6m"] = df["events_prev_year"] / 2
    df["events_12m"] = df["events_prev_year"]
    df["events_24m"] = df["events_prev_year"] * 2
    df["days_since_last"] = 30  # placeholder

    # Save live features
    os.makedirs("data/live", exist_ok=True)
    df.to_csv(LIVE_FEATURES, index=False)
    print(f"✅ Wrote live features → {LIVE_FEATURES} (rows={len(df)})")

if __name__ == "__main__":
    build_live_features()
