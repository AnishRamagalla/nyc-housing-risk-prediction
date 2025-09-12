# live/ingest_complaints.py
import os
import pandas as pd
from datetime import datetime

# Paths
SRC = "data/processed/building_features.csv"
OUT = "data/live/features.csv"

os.makedirs("data/live", exist_ok=True)

def main():
    if not os.path.exists(SRC):
        print(f"❌ Source file not found: {SRC}")
        return

    df = pd.read_csv(SRC)

    # Add a snapshot_date column for freshness
    df["snapshot_date"] = pd.to_datetime(df["snapshot_year"], format="%Y", errors="coerce")
    df["snapshot_date"] = df["snapshot_date"].fillna(datetime.utcnow().date())

    # Save to live features
    df.to_csv(OUT, index=False)
    print(f"✅ Wrote live features → {OUT} (rows={len(df)})")

if __name__ == "__main__":
    main()
