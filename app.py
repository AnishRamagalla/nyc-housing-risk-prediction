import os
import joblib
import shap
import folium
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "models/monthly_rf_v1.joblib"
FEATURES_FILE = "data/live/features.csv"

# -----------------------------
# Geocoder (OpenStreetMap / Nominatim)
# -----------------------------
geolocator = Nominatim(user_agent="nyc-housing-risk")

def address_to_location(address: str):
    """Convert free-text address to lat/lon + display name."""
    try:
        loc = geolocator.geocode(address)
        if loc:
            return {
                "lat": loc.latitude,
                "lon": loc.longitude,
                "display_name": loc.address
            }
        else:
            return None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None

# -----------------------------
# Load model and features
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Run training first.")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    if not os.path.exists(FEATURES_FILE):
        st.error("❌ Features file not found. Run src/features.py first.")
        return None
    return pd.read_csv(FEATURES_FILE, parse_dates=["snapshot_date"])

def align_to_model(X, model):
    need = list(model.feature_names_in_)
    X = X.reindex(columns=need, fill_value=0)
    return X[need]

# -----------------------------
# Prediction + SHAP
# -----------------------------
def predict_with_explain_for_bin(bin_value, model, features_df):
    row = features_df[features_df["building_id"].astype(str) == str(bin_value)]
    if row.empty:
        return None, None

    # Features only
    X = row.drop(columns=["building_id", "snapshot_date", "latitude", "longitude"], errors="ignore")
    X = align_to_model(X, model)

    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    # SHAP
    explanation = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv_raw = shap_values[1][0]
        else:
            sv_raw = shap_values[0]

        sv = np.ravel(np.array(sv_raw, dtype="float64"))
        vals = np.ravel(np.array(X.iloc[0].values, dtype="float64"))
        feats = list(model.feature_names_in_)

        if sv.shape[0] != len(feats):
            sv = sv[: len(feats)]
        if vals.shape[0] != len(feats):
            vals = vals[: len(feats)]

        explanation = (
            pd.DataFrame({"feature": feats, "shap_value": sv, "value": vals})
            .sort_values("shap_value", key=np.abs, ascending=False)
        )
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    result = {
        "building_id": str(bin_value),
        "snapshot": str(row["snapshot_date"].iloc[0]),
        "risk_prob": prob,
        "pred_label": pred,
    }
    return result, explanation

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NYC Housing Risk", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

model = load_model()
features_df = load_features()

address = st.text_input("Enter Address (e.g., 124 Lake St, Brooklyn, NY)")

if st.button("Predict Risk"):
    if not model or features_df is None:
        st.stop()

    # Step 1: Geocode
    location = address_to_location(address)
    if not location:
        st.error("Could not geocode that address. Try another one.")
        st.stop()

    st.success(f"📍 Found: {location['display_name']}")

    # Step 2: Map
    fmap = folium.Map(location=[location["lat"], location["lon"]], zoom_start=16)
    folium.Marker(
        [location["lat"], location["lon"]],
        popup=location["display_name"],
        tooltip="Input Address",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(fmap)
    st.subheader("Location")
    st_folium(fmap, width=750, height=500)

    # Step 3: Try to match BIN
    # (Right now, your features.csv needs to have BIN for this to work properly)
    # For demo, let’s just use the first BIN in dataset
    bin_guess = features_df["building_id"].iloc[0]

    result, explanation = predict_with_explain_for_bin(bin_guess, model, features_df)

    if result:
        st.subheader("Prediction Result")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Building ID (BIN)", result["building_id"])
        with c2:
            st.metric("Risk Probability", f"{result['risk_prob']:.3f}")
        with c3:
            label = "High Risk" if result["pred_label"] else "Low Risk"
            st.metric("Predicted Label", label)
        st.caption(f"Snapshot: {result['snapshot']}")

        if explanation is not None and not explanation.empty:
            st.subheader("Top Features Driving Risk")
            st.dataframe(explanation.head(8))
            st.bar_chart(explanation.set_index("feature")["shap_value"].head(10))
    else:
        st.warning("No prediction available for this address (BIN not found in dataset).")
