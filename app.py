import streamlit as st
import pandas as pd
import joblib
import os
import requests
import shap
import folium
from streamlit_folium import st_folium

# ---------------- CONFIG ----------------
MODEL_PATH = "models/monthly_rf_v1.joblib"
FEATURES_FILE = "data/processed/features_monthly.csv"

st.set_page_config(page_title="NYC Housing Risk", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

# Debug: check if secrets are loaded
st.caption(f"Secrets loaded? {'MAPBOX_TOKEN' in st.secrets}")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run training first.")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
feature_list = None
if model:
    if hasattr(model, "feature_names_in_"):
        feature_list = list(model.feature_names_in_)

# ---------------- SAFE GEOCODING ----------------
def geocode_address(address):
    """Try Mapbox, fallback to OpenStreetMap Nominatim."""
    try:
        token = st.secrets.get("MAPBOX_TOKEN", None)

        # 1. Mapbox geocoder
        if token:
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
            params = {"access_token": token, "limit": 1}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("features"):
                    coords = data["features"][0]["geometry"]["coordinates"]
                    lon, lat = coords
                    place = data["features"][0]["place_name"]
                    return lat, lon, place
            st.warning("⚠️ Mapbox could not geocode this address. Trying OSM...")

        # 2. Fallback: Nominatim (OpenStreetMap)
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json", "limit": 1}
        r = requests.get(url, params=params, headers={"User-Agent": "nyc-housing-risk-app"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                place = data[0]["display_name"]
                return lat, lon, place

        return None, None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None, None

# ---------------- ALIGN COLUMNS ----------------
def align_columns(df, model):
    need = list(model.feature_names_in_)
    for c in need:
        if c not in df.columns:
            df[c] = 0.0
    extra = [c for c in df.columns if c not in need]
    if extra:
        df = df.drop(columns=extra)
    return df[need]

# ---------------- PREDICT WITH EXPLANATION ----------------
def predict_with_explain(building_id, model):
    if not os.path.exists(FEATURES_FILE):
        st.error("Features file not found. Run feature engineering first.")
        return None, None

    df = pd.read_csv(FEATURES_FILE, parse_dates=["snapshot_date"])
    row = df[df["bbl"].astype(str) == str(building_id)]  # using BBL here
    if row.empty:
        st.warning(f"No features found for building {building_id}")
        return None, None

    X = row.drop(columns=["bbl", "snapshot_date"], errors="ignore")
    X = align_columns(X, model)

    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    # SHAP Explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_for_class1 = shap_values[1][0]
        else:
            shap_for_class1 = shap_values[0]

        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "shap_value": shap_for_class1,
            "value": X.iloc[0].values
        }).sort_values("shap_value", key=abs, ascending=False)

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
        feature_importance = None

    result = {
        "building_id": building_id,
        "snapshot": row["snapshot_date"].iloc[0],
        "risk_prob": prob,
        "pred_label": pred
    }
    return result, feature_importance

# ---------------- UI ----------------
address = st.text_input("Enter Address (e.g., 124 Lake St, Brooklyn, NY)", "124 Lake St, Brooklyn, NY")

if st.button("Predict Risk"):
    if not model:
        st.error("No model loaded.")
    else:
        # Geocode
        lat, lon, place = geocode_address(address)

        if lat and lon:
            st.success(f"📍 Located: {place}")
            m = folium.Map(location=[lat, lon], zoom_start=15)
            folium.Marker([lat, lon], tooltip=place).add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.info("No location found for this address.")

        # For now we’ll use BBL lookup = fake key from address
        building_id = hash(address) % 10000000  # ⚠️ replace with real BBL mapping if available

        result, explanation = predict_with_explain(building_id, model)
        if result:
            st.subheader("Prediction Result")
            st.write(f"**Building ID (BBL):** {result['building_id']}")
            st.write(f"**Snapshot Date:** {result['snapshot']}")
            st.write(f"**Risk Probability:** {result['risk_prob']:.3f}")
            st.write(f"**Predicted Label:** {result['pred_label']}")

            if explanation is not None:
                st.subheader("Top Features Driving Risk")
                st.dataframe(explanation.head(5))
                st.bar_chart(explanation.set_index("feature")["shap_value"].head(10))
            else:
                st.info("Model explanation not available.")
