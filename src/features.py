import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

DB_PATH = "data/live/complaints.sqlite"
FEATURES_OUT = "data/live/features.csv"

def load_complaints():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT bin, date_entered FROM complaints", conn)
    conn.close()
    # Ensure datetime
    df["date_entered"] = pd.to_datetime(df["date_entered"], errors="coerce")
    df = df.dropna(subset=["bin","date_entered"])
    df["bin"] = df["bin"].astype(str).str.strip()
    return df

def build_features(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df["date_entered"].max()
    out = []
    for b, group in df.groupby("bin"):
        last_date = group["date_entered"].max()
        feats = {
            "building_id": b,
            "snapshot_date": snapshot_date,
            "days_since_last": (snapshot_date - last_date).days,
            "events_3m": (group["date_entered"] >= snapshot_date - timedelta(days=90)).sum(),
            "events_6m": (group["date_entered"] >= snapshot_date - timedelta(days=180)).sum(),
            "events_12m": (group["date_entered"] >= snapshot_date - timedelta(days=365)).sum(),
            "events_24m": (group["date_entered"] >= snapshot_date - timedelta(days=730)).sum(),
        }
        out.append(feats)
    return pd.DataFrame(out)

def update_features():
    df = load_complaints()
    if df.empty:
        print("No complaints available.")
        return
    snapshot_date = df["date_entered"].max()
    features = build_features(df, snapshot_date=snapshot_date)
    os.makedirs("data/live", exist_ok=True)
    features.to_csv(FEATURES_OUT, index=False)
    print(f"Saved features for {len(features)} buildings at snapshot {snapshot_date}")

if __name__ == "__main__":
    update_features()
