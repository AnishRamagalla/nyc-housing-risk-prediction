import os, requests, datetime
import pandas as pd

BASE = "https://data.cityofnewyork.us/resource"
DATASET = "vztk-gaf7"
APP_TOKEN = os.getenv("SODA_APP_TOKEN")

params = {
    "$limit": 20,
    "$order": "dobrundate DESC"   # most recent complaints
}
headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}

url = f"{BASE}/{DATASET}.json"
r = requests.get(url, params=params, headers=headers, timeout=30)
r.raise_for_status()
rows = r.json()

print(f"Token found? {'YES' if APP_TOKEN else 'NO'}")
print(f"Fetched {len(rows)} rows")

# Convert to DataFrame for easy viewing
df = pd.DataFrame(rows)
print(df.head())

# Convert date_entered to datetime
df["date_entered"] = pd.to_datetime(df["date_entered"], errors="coerce")
recent = df[df["date_entered"] >= (datetime.datetime.utcnow() - datetime.timedelta(days=7))]
print("\nComplaints in last 7 days:")
print(recent[["bin","date_entered","house_number","house_street","complaint_category"]].head())
