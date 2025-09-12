import streamlit as st
import pandas as pd
import joblib
import requests
import os
import shap
import folium
from streamlit_folium import st_folium

# --- Config ---
MODEL_PATH = "models/monthly_rf_v1.joblib"
FEATURES_FILE = "data/processed/features_monthly.csv"

# --- Load model ---
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("Model file not found. Train first.")
            return None
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# --- Geocode Address ---
def geocode_address(address):
    token = st.secrets.get("MAPBOX_TOKEN", None)
    if not token:
        st.error("⚠️ MAPBOX_TOKEN missing in secrets.toml or Streamlit Cloud settings.")
        return None, None, None

    try:
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
        params = {"access_token": token, "limit": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("features"):
            coords = data["features"][0]["geometry"]["coordinates"]
            lon, lat = coords
            place = data["features"][0]["place_name"]
            return lat, lon, place
        else:
            st.warning("⚠️ No geocoding results found.")
    except Exception as e:
        st.warning(f"⚠️ Geocoding failed: {e}")
    return None, None, None

# --- Align columns ---
def align_columns(df, model):
    try:
        need = list(model.feature_names_in_)
        for c in need:
            if c not in df.columns:
                df[c] = 0.0
        extra = [c for c in df.columns if c not in need]
        if extra:
            df = df.drop(columns=extra)
        return df[need]
    except Exception as e:
        st.warning(f"⚠️ Column alignment failed: {e}")
        return df

# --- Predict with SHAP ---
def predict_with_explain(row, model):
    try:
        X = row.drop(columns=["snapshot_date"], errors="ignore")
        X = align_columns(X, model)
        prob = model.predict_proba(X)[0, 1]
        pred = int(prob >= 0.5)

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_for_class1 = shap_values[1]
        else:
            shap_for_class1 = shap_values
        shap_for_class1 = shap_for_class1[0]

        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "shap_value": shap_for_class1,
            "value": X.iloc[0].values
        }).sort_values("shap_value", key=abs, ascending=False)

        return {"risk_prob": prob, "pred_label": pred}, feature_importance
    except Exception as e:
        st.warning(f"⚠️ Prediction/SHAP failed: {e}")
        return {"risk_prob": 0, "pred_label": 0}, None

# --- Streamlit UI ---
st.set_page_config(page_title="NYC Housing Risk", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

model = load_model()
address = st.text_input("Enter Address (e.g., 124 Lake St, Brooklyn, NY)", "124 Lake St, Brooklyn, NY")

if st.button("Predict Risk"):
    if not model:
        st.stop()

    lat, lon, place = geocode_address(address)
    if not lat:
        st.stop()

    st.success(f"📍 Address found: {place}")
    st.write(f"Coordinates: ({lat}, {lon})")

    if not os.path.exists(FEATURES_FILE):
        st.error("❌ Features file not found. Run feature generation first.")
        st.stop()

    try:
        df = pd.read_csv(FEATURES_FILE, parse_dates=["snapshot_date"])
        last = df["snapshot_date"].max()
        row = df[df["snapshot_date"] == last].sample(1, random_state=42)

        result, explanation = predict_with_explain(row, model)

        st.subheader("Prediction Result")
        st.write(f"**Risk Probability:** {result['risk_prob']:.3f}")
        st.write(f"**Predicted Label:** {result['pred_label']}")

        if explanation is not None:
            st.subheader("Top Features Driving Risk")
            st.dataframe(explanation.head(5))
            st.bar_chart(explanation.set_index("feature")["shap_value"].head(10))

        # Map
        st.subheader("Building Location")
        m = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker([lat, lon], popup=place, tooltip="Building").add_to(m)
        st_folium(m, width=700, height=500)

    except Exception as e:
        st.error(f"❌ Failed to run prediction pipeline: {e}")
