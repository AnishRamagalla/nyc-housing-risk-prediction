import streamlit as st
import pandas as pd
import joblib
import os
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# --- Config ---
MODEL_PATH = "models/monthly_rf_v1.joblib"
FEATURES_FILE = "data/processed/features_monthly.csv"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

st.set_page_config(page_title="NYC Housing Risk", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

# --- Input ---
address = st.text_input("Enter Address (e.g., 124 Lake St, Brooklyn, NY)", "")

if st.button("Predict Risk"):
    if not address.strip():
        st.warning("Please enter an address")
    else:
        # --- Geocode ---
        geolocator = Nominatim(user_agent="nyc-housing-risk")
        location = geolocator.geocode(address)

        if location:
            st.success(f"📍 Found: {location.address}")
            st.write(f"Latitude: {location.latitude}, Longitude: {location.longitude}")

            # --- Show map ---
            m = folium.Map(location=[location.latitude, location.longitude], zoom_start=16)
            folium.Marker([location.latitude, location.longitude], tooltip=address).add_to(m)
            st_folium(m, width=700, height=500)

        else:
            st.error("Address not found. Try with more details (e.g., borough).")
