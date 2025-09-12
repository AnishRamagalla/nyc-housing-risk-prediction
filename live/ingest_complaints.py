import os, requests, sqlite3, datetime, pandas as pd

BASE = "https://data.cityofnewyork.us/resource"
DATASET = "vztk-gaf7"
APP_TOKEN = os.getenv("SODA_APP_TOKEN")

DB_PATH = "data/live/complaints.sqlite"

def fetch_recent(limit=100):
    params = {
        "$limit": limit,
        "$order": "dobrundate DESC"
    }
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    url = f"{BASE}/{DATASET}.json"
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def init_db():
    os.makedirs("data/live", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            complaint_number TEXT PRIMARY KEY,
            bin TEXT,
            date_entered TEXT,
            house_number TEXT,
            house_street TEXT,
            complaint_category TEXT,
            status TEXT,
            inspection_date TEXT,
            dobrundate TEXT
        )
    """)
    conn.commit()
    return conn

def save_to_db(df):
    conn = init_db()
    df.to_sql("complaints", conn, if_exists="append", index=False)
    conn.close()

def ingest():
    df = fetch_recent(limit=200)
    if df.empty:
        print("No data fetched.")
        return
    # Keep only relevant columns
    keep = ["complaint_number","bin","date_entered","house_number",
            "house_street","complaint_category","status",
            "inspection_date","dobrundate"]
    df = df[keep]
    # Convert date_entered to standard ISO format
    df["date_entered"] = pd.to_datetime(df["date_entered"], errors="coerce").astype(str)
    # Drop duplicates if already in DB
    conn = init_db()
    existing = pd.read_sql("SELECT complaint_number FROM complaints", conn)
    new = df[~df["complaint_number"].isin(existing["complaint_number"])]
    if not new.empty:
        new.to_sql("complaints", conn, if_exists="append", index=False)
        print(f"Inserted {len(new)} new complaints.")
    else:
        print("No new complaints to insert.")
    conn.close()

if __name__ == "__main__":
    ingest()
