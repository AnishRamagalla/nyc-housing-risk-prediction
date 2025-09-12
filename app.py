import streamlit as st
import pandas as pd
import joblib
import json
import folium
from streamlit_folium import st_folium
import requests
import os

# ---------------- PATHS ----------------
MODEL_PATH = "artifacts/model.pkl"
FEATURE_LIST_PATH = "artifacts/feature_list.json"
FEATURES_FILE = "data/processed/building_features.csv"

# ---------------- LOAD MODEL + FEATURES ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Run Notebook 3 to train and save model.")
        return None, None
    model = joblib.load(MODEL_PATH)

    if not os.path.exists(FEATURE_LIST_PATH):
        st.error("❌ Feature list not found. Run Notebook 3.")
        return None, None

    with open(FEATURE_LIST_PATH, "r") as f:
        feature_list = json.load(f)

    return model, feature_list

# ---------------- GEOCODING ----------------
def geocode_address(address: str):
    """Use Mapbox API to geocode an address -> lat, lon"""
    token = st.secrets.get("MAPBOX_TOKEN")
    if not token:
        st.error("❌ MAPBOX_TOKEN missing. Add it to .streamlit/secrets.toml")
        return None, None, None

    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
    params = {"access_token": token, "limit": 1}
    resp = requests.get(url, params=params)

    if resp.status_code != 200:
        st.error(f"❌ Geocoding API failed: {resp.text}")
        return None, None, None

    data = resp.json()
    if not data.get("features"):
        st.warning("⚠️ No geocoding results found.")
        return None, None, None

    coords = data["features"][0]["center"]
    lon, lat = coords
    place_name = data["features"][0]["place_name"]
    return lat, lon, place_name

# ---------------- PREDICTION ----------------
def predict(building_id, model, feature_list):
    df = pd.read_csv(FEATURES_FILE)
    row = df[df["bbl"].astype(str) == str(building_id)]

    if row.empty:
        return None, None

    X = row.reindex(columns=feature_list, fill_value=0)
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    return {
        "building_id": building_id,
        "risk_prob": float(prob),
        "pred_label": pred
    }, X.iloc[0].to_dict()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="🏠 NYC Housing Risk", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

# Load model
model, feature_list = load_model()

# Address input
address = st.text_input("Enter address (e.g., 124 Lake St, Brooklyn, NY)")

if st.button("🔍 Predict"):
    if not model:
        st.stop()

    lat, lon, place = geocode_address(address)
    if lat is None:
        st.stop()

    # Display map
    st.subheader("📍 Location")
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], popup=place).add_to(m)
    st_folium(m, width=700, height=500)

    # Try prediction
    building_id = st.text_input("Enter building BBL (from dataset)", "")
    if building_id:
        result, features_used = predict(building_id, model, feature_list)

        if result:
            st.subheader("📊 Prediction Result")
            st.write(f"**Building ID:** {result['building_id']}")
            st.write(f"**Risk Probability:** {result['risk_prob']:.3f}")
            st.write(f"**Predicted Label:** {result['pred_label']}")

            st.subheader("🔎 Features Used")
            st.json(features_used)
        else:
            st.warning("⚠️ No matching building found in dataset.")
