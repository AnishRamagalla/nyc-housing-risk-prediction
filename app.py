import streamlit as st
import pandas as pd
import joblib
import os
import requests
import folium
from streamlit_folium import st_folium

# --- Config ---
MODEL_PATH = "models/monthly_rf_v1.joblib"
FEATURES_FILE = "data/processed/features_monthly.csv"

# --- Load model + features ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found.")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    if not os.path.exists(FEATURES_FILE):
        st.error("❌ Features file not found.")
        return None
    df = pd.read_csv(FEATURES_FILE, parse_dates=["snapshot_date"])
    return df

# --- Geocode using NYC GeoSearch ---
def geocode_address(address):
    url = "https://geosearch.planning.nyc.gov/v1/search"
    params = {"text": address, "size": 1}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        if data.get("features"):
            coords = data["features"][0]["geometry"]["coordinates"]
            lon, lat = coords
            props = data["features"][0]["properties"]
            return lat, lon, props.get("bbl"), props.get("bin"), props
    return None, None, None, None, None

# --- Streamlit UI ---
st.set_page_config(page_title="NYC Housing Risk", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

address = st.text_input("Enter Address (e.g., 124 Lake St, Brooklyn, NY)", "")

model = load_model()
features = load_features()

if st.button("Predict Risk") and address:
    lat, lon, bbl, bin_id, props = geocode_address(address)

    if lat and lon:
        # --- Map ---
        m = folium.Map(location=[lat, lon], zoom_start=16)
        folium.Marker([lat, lon], popup=address, tooltip="Input Address", icon=folium.Icon(color="blue")).add_to(m)
        st_folium(m, width=700, height=400)

        st.success(f"📍 Found BIN: {bin_id}, BBL: {bbl}")
        st.json(props)  # show raw metadata for now

        # --- Prediction ---
        if model is not None and features is not None:
            last_date = features["snapshot_date"].max()
            row = features[(features["snapshot_date"] == last_date) & (features["building_id"].astype(str) == str(bin_id))]

            if row.empty:
                st.warning("⚠️ No features found for this building in dataset.")
            else:
                X_sample = row.drop(columns=["building_id", "snapshot_date", "label"], errors="ignore").fillna(0)
                y_prob = model.predict_proba(X_sample)[0, 1]
                y_pred = int(y_prob >= 0.5)

                st.subheader("Prediction Result")
                st.write(f"🏢 **Building ID (BIN):** {bin_id}")
                st.write(f"📅 **Snapshot Date:** {last_date}")
                st.write(f"🔥 **Risk Probability:** {y_prob:.3f}")
                st.write(f"✅ **Predicted Label:** {y_pred}")
    else:
        st.error("❌ Could not geocode this address. Try with more details.")
