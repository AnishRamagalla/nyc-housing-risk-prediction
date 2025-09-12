import os
import joblib
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# ---------------- SETTINGS ----------------
MODEL_PATH = "artifacts/model.pkl"
FEATURES_PATH = "data/processed/building_features.csv"

# Mapbox token
MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]

# ---------------- HELPERS ----------------
def geocode_address(address: str):
    """Geocode an address into lat/lon using Mapbox API"""
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
    params = {"access_token": MAPBOX_TOKEN, "limit": 1}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        if data.get("features"):
            coords = data["features"][0]["geometry"]["coordinates"]
            lon, lat = coords
            place = data["features"][0]["place_name"]
            return lat, lon, place
    return None, None, None

def load_model_and_features():
    """Load trained model and features dataset"""
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file missing. Please retrain.")
        st.stop()
    if not os.path.exists(FEATURES_PATH):
        st.error("❌ Features file missing. Please preprocess.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH)

    # Keep only latest snapshot
    if "snapshot_year" in df.columns:
        latest = df["snapshot_year"].max()
        df = df[df["snapshot_year"] == latest]

    return model, df

def prepare_features(df, building_id, feature_cols):
    """Extract row for building_id; fallback to unknown"""
    if building_id in df["bbl"].astype(str).values:
        row = df[df["bbl"].astype(str) == str(building_id)].copy()
    else:
        st.warning("⚠️ No exact match, using 'unknown'.")
        row = df[df["bbl"].astype(str) == "unknown"].copy()

    # Drop non-features
    drop_cols = ["bbl", "bin", "address", "snapshot_year", "label"]
    row = row.drop(columns=[c for c in drop_cols if c in row.columns])

    # Align with training features
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]

    return row

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="NYC Housing Risk", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

address = st.text_input("Enter Address (or BIN/BBL):", "124 Lake St, Brooklyn, NY")

if st.button("Predict Risk"):
    lat, lon, place = geocode_address(address)
    if not lat:
        st.error("❌ Could not geocode address.")
        st.stop()

    st.success(f"📍 Location found: {place} (lat={lat}, lon={lon})")

    # Load model + features
    model, df = load_model_and_features()

    # Training feature list
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = df.drop(columns=["bbl", "bin", "address", "snapshot_year", "label"], errors="ignore").columns.tolist()

    # Pick a building ID (simulate with bbl or fallback)
    building_id = address if address.isdigit() else "unknown"
    X_input = prepare_features(df, building_id, feature_cols)

    try:
        pred = model.predict(X_input)[0]
        st.success(f"✅ Prediction for **{place}**: {int(pred)}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

    # Show map
    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], popup=place).add_to(m)
    st_folium(m, width=700, height=400)
