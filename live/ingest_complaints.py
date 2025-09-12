# live/ingest_complaints.py
import os
import sys
import time
import json
import math
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

LIVE_DIR = "data/live"
LIVE_FEATURES = os.path.join(LIVE_DIR, "features.csv")
FALLBACK_EVENTS = "data/processed/events_sample_for_debug.csv"  # fallback if API unavailable

# NYC Open Data: DOB Complaints (dataset commonly exposed as 'ipu4-2q9a')
NYC_DATASET = "ipu4-2q9a"
NYC_BASE = f"https://data.cityofnewyork.us/resource/{NYC_DATASET}.json"

APP_TOKEN = os.environ.get("NYC_APP_TOKEN")  # optional but recommended

def fetch_dob_complaints(since_days=730, limit=50000):
    """Fetch DOB complaints with BBL + dates, last N days."""
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    params = {
        "$select": "bbl,received_date",
        "$where": f"received_date >= '{since}' AND bbl IS NOT NULL",
        "$limit": limit,
        "$order": "received_date DESC"
    }
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}

    try:
        r = requests.get(NYC_BASE, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        # Normalize
        df = df.rename(columns={"received_date": "created_date"})
        df["created_date"] = pd.to_datetime(df["created_date"], utc=True, errors="coerce")
        df = df.dropna(subset=["bbl", "created_date"])
        df["bbl"] = df["bbl"].astype(str)
        return df[["bbl", "created_date"]]
    except Exception as e:
        print(f"API fetch failed ({e}). Will try local fallback.", file=sys.stderr)
        return None

def load_events_fallback():
    if not os.path.exists(FALLBACK_EVENTS):
        return None
    df = pd.read_csv(FALLBACK_EVENTS)
    # Expecting at least: bbl, created_date (or snapshot_year to fake dates)
    if "created_date" not in df.columns:
        # fabricate dates using snapshot_year if present
        if "snapshot_year" in df.columns:
            df["created_date"] = pd.to_datetime(df["snapshot_year"].astype(str) + "-12-31")
        else:
            df["created_date"] = pd.to_datetime("today")
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["bbl"] = df["bbl"].astype(str)
    return df[["bbl", "created_date"]]

def build_features_from_events(df_events):
    """Aggregate events into features expected by model."""
    now = pd.Timestamp.utcnow()
    df = df_events.copy()
    df["created_date"] = pd.to_datetime(df["created_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_date"])

    def win(days):
        return df[df["created_date"] >= (now - pd.Timedelta(days=days))].groupby("bbl").size()

    f_3m  = win(90).rename("events_3m")
    f_6m  = win(180).rename("events_6m")
    f_12m = win(365).rename("events_12m")
    f_24m = win(730).rename("events_24m")

    last = df.groupby("bbl")["created_date"].max().rename("last_event")
    days_since = ((now - last).dt.days).rename("days_since_last")

    features = pd.concat([f_3m, f_6m, f_12m, f_24m, days_since], axis=1).fillna(0).reset_index()
    features["days_since_last"] = features["days_since_last"].astype(int)
    features["snapshot_date"] = now.normalize()
    return features

def main():
    os.makedirs(LIVE_DIR, exist_ok=True)

    events = fetch_dob_complaints(since_days=730, limit=100000)
    if events is None:
        events = load_events_fallback()
        if events is None:
            print(f"No events available (API and fallback both missing).", file=sys.stderr)
            sys.exit(2)

    features = build_features_from_events(events)
    # Ensure bbl column is string
    features["bbl"] = features["bbl"].astype(str)

    # Keep only columns used by the model + bbl
    need_cols = ["bbl", "events_3m", "events_6m", "events_12m", "events_24m", "days_since_last", "snapshot_date"]
    for col in need_cols:
        if col not in features.columns:
            features[col] = 0

    features = features[need_cols]
    features.to_csv(LIVE_FEATURES, index=False)
    print(f"✅ Wrote live features → {LIVE_FEATURES} (rows={len(features)})")

if __name__ == "__main__":
    main()
