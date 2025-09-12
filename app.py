import os
import requests
import joblib
import pandas as pd
import shap
import streamlit as st
import folium
from streamlit_folium import st_folium

# ---------------- SETTINGS ----------------
MODEL_PATH = "artifacts/model.pkl"
FEATURES_PATH = "data/processed/features_monthly.csv"

# ---------------- HELPERS -----------------
def geocode_address(address):
    """Geocode using Mapbox API"""
    token = st.secrets["MAPBOX_TOKEN"]
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
    params = {"access_token": token, "limit": 1}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        if data.get("features"):
            feat = data["features"][0]
            lon, lat = feat["geometry"]["coordinates"]
            place = feat["place_name"]
            return lat, lon, place
    return None, None, None


def predict_with_explain(bid, model, feature_list, df):
    """Run prediction + SHAP explanation safely"""
    sample = df[df["bbl"] == bid]
    if sample.empty:
        return None, "No data for this building."

    X_sample = sample[feature_list].fillna(0)
    prob = model.predict_proba(X_sample)[0, 1]
    pred = int(prob >= 0.5)

    explanation = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle tree-based models (list of arrays)
        if isinstance(shap_values, list):
            shap_for_class1 = shap_values[1][0]
        else:
            shap_for_class1 = shap_values[0]

        # Align features + shap values
        n = min(len(feature_list), len(shap_for_class1))
        shap_series = pd.Series(shap_for_class1[:n], index=feature_list[:n])

        explanation = (
            pd.DataFrame({
                "feature": shap_series.index,
                "shap_value": shap_series.values,
                "value": X_sample.iloc[0, :n].values
            })
            .sort_values(by="shap_value", key=abs, ascending=False)
            .head(10)
        )
    except Exception as e:
        explanation = f"SHAP explanation unavailable: {e}"

    return {"prob": prob, "pred": pred}, explanation

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="NYC Housing Risk Prediction", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

# Debug: secrets loaded
st.write("Secrets loaded?", "✅" if "MAPBOX_TOKEN" in st.secrets else "❌")

# Debug: paths
with st.expander("🔍 Debug paths"):
    st.write("cwd:", os.getcwd())
    st.write("Model exists?", os.path.exists(MODEL_PATH))
    st.write("Features exists?", os.path.exists(FEATURES_PATH))

# Load data & model
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
df = pd.read_csv(FEATURES_PATH, parse_dates=["snapshot_date"]) if os.path.exists(FEATURES_PATH) else None
feature_list = [c for c in df.columns if c not in ("bbl", "label", "snapshot_date")] if df is not None else []

# User input
address = st.text_input("Enter Address (e.g., 124 Lake St, Brooklyn, NY)", "")
if st.button("Predict Risk"):
    lat, lon, place = geocode_address(address)

    if not lat:
        st.error("❌ Could not geocode the address. Try another one.")
    else:
        st.success(f"📍 Location found: {place} (lat={lat}, lon={lon})")

        if model is None or df is None:
            st.error("❌ Model or features file missing.")
        else:
            # Try mapping building ID from address (mock: pick random BBL)
            if "bbl" in df.columns:
                building_id = df.sample(1)["bbl"].iloc[0]
            else:
                building_id = 0

            result, explanation = predict_with_explain(building_id, model, feature_list, df)

            if result:
                st.subheader("📊 Prediction Result")
                st.write(f"**Building ID:** {building_id}")
                st.write(f"**Risk Probability:** {result['prob']:.3f}")
                st.write(f"**Predicted Label:** {result['pred']}")

            # SHAP explanation
            st.subheader("🔎 Model Explanation")
            if isinstance(explanation, pd.DataFrame):
                st.dataframe(explanation)
                st.bar_chart(explanation.set_index("feature")["shap_value"])
            else:
                st.info(explanation)

            # Show map
            st.subheader("🗺️ Map")
            m = folium.Map(location=[lat, lon], zoom_start=15)
            folium.Marker([lat, lon], popup=place).add_to(m)
            st_folium(m, width=700, height=500)
