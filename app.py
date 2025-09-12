# app.py — crash-proof, SHAP-safe, with debug info
import os
import json
import numpy as np
import pandas as pd
import requests
import joblib
import streamlit as st
import folium
from streamlit_folium import st_folium

# -------------------- SETTINGS --------------------
st.set_page_config(page_title="NYC Housing Risk", layout="wide")
st.set_option("client.showErrorDetails", True)

MODEL_PATH = "models/monthly_rf_v1.joblib"
FEATURES_FILE = "data/processed/features_monthly.csv"

# -------------------- HELPERS ---------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model file not found: models/monthly_rf_v1.joblib")
            return None
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

@st.cache_data
def load_features_df():
    try:
        if not os.path.exists(FEATURES_FILE):
            st.error("❌ Features file not found: data/processed/features_monthly.csv")
            return None
        df = pd.read_csv(FEATURES_FILE, parse_dates=["snapshot_date"])
        return df
    except Exception as e:
        st.error(f"❌ Failed to load features: {e}")
        return None

def geocode_address(address: str):
    """Geocode with Mapbox. Returns (lat, lon, place) or (None, None, None)."""
    token = st.secrets.get("MAPBOX_TOKEN", None)
    if not token:
        st.error("❌ MAPBOX_TOKEN missing in secrets. Add it in Streamlit Cloud → Settings → Secrets.")
        return None, None, None

    try:
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
        r = requests.get(url, params={"access_token": token, "limit": 1}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("features"):
            f = data["features"][0]
            lon, lat = f["center"]
            return lat, lon, f.get("place_name", address)
    except Exception as e:
        st.warning(f"⚠️ Geocoding failed: {e}")
    return None, None, None

def align_columns_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Ensure X has exactly model.feature_names_in_ columns (add missing=0, drop extras)."""
    try:
        need = list(model.feature_names_in_)
    except Exception:
        # Fallback: if model has no feature_names_in_ (unlikely for sklearn>=1.0)
        need = list(X.columns)

    # Add missing
    for c in need:
        if c not in X.columns:
            X[c] = 0.0
    # Drop extras
    extras = [c for c in X.columns if c not in need]
    if extras:
        X = X.drop(columns=extras)
    # Order
    return X[need]

def safe_shap_values(model, X):
    """
    Compute SHAP values robustly across shap versions.
    Returns a 1D numpy array of length n_features for the first row, or None on failure.
    """
    try:
        import shap  # import here to avoid import-time crashes breaking the whole app
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)

        # shap>=0.48 sometimes returns Explanation object
        if hasattr(sv, "values"):   # Explanation
            values = sv.values
        else:
            values = sv

        # Binary classifier may return list [class0, class1]
        if isinstance(values, list):
            # Prefer class 1 if available
            arr = values[1] if len(values) > 1 else values[0]
        else:
            arr = values

        arr = np.array(arr)
        # Expect shape (n_samples, n_features) → take first row
        if arr.ndim == 2:
            row_vals = arr[0]
        elif arr.ndim == 1:
            row_vals = arr
        else:
            return None

        return np.array(row_vals, dtype=float).ravel()
    except Exception as e:
        st.warning(f"⚠️ SHAP unavailable: {e}")
        return None

def build_feature_importance_df(X_row: pd.Series, shap_1d: np.ndarray) -> pd.DataFrame:
    """
    Build a tidy feature-importance DF from one sample row and a 1D shap array.
    Clips to the minimum common length to avoid shape errors.
    """
    cols = list(X_row.index)
    vals = np.array(X_row.values, dtype=float).ravel()

    n = min(len(cols), len(vals), len(shap_1d))
    rows = []
    for i in range(n):
        rows.append({
            "feature": cols[i],
            "value": vals[i],
            "shap_value": float(shap_1d[i]),
        })
    fi = pd.DataFrame(rows)
    if not fi.empty:
        fi = fi.reindex(fi["shap_value"].abs().sort_values(ascending=False).index)
    return fi

# -------------------- UI ---------------------
st.title("🏠 NYC Housing Risk Prediction")
st.caption(f"Secrets loaded? {'✅' if 'MAPBOX_TOKEN' in st.secrets else '❌'}")

with st.expander("Debug paths"):
    st.write(f"cwd: {os.getcwd()}")
    st.write(f"Model exists? {os.path.exists(MODEL_PATH)}")
    st.write(f"Features exists? {os.path.exists(FEATURES_FILE)}")

address = st.text_input("Enter Address (e.g., 124 Lake St, Brooklyn, NY)", "124 Lake St, Brooklyn, NY")

if st.button("Predict Risk"):
    # 1) Geocode
    lat, lon, place = geocode_address(address)
    if not lat:
        st.stop()

    st.success(f"📍 {place}")
    # Map
    try:
        m = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker([lat, lon], tooltip=place, icon=folium.Icon(color="red")).add_to(m)
        st_folium(m, width=700, height=500)
    except Exception as e:
        st.warning(f"⚠️ Map render failed: {e}")

    # 2) Load model + features
    model = load_model()
    df = load_features_df()
    if model is None or df is None:
        st.stop()

    # 3) Pick a building row to score (until exact BBL mapping is implemented)
    try:
        # Use latest snapshot; sample one building to guarantee a row
        last = df["snapshot_date"].max()
        sample_row = df[df["snapshot_date"] == last].sample(1, random_state=42).copy()
        # Use BBL as building_id if present; else synthesize one
        building_id = str(sample_row.get("bbl", pd.Series(["unknown"])).iloc[0])

        # Prepare X
        X = sample_row.drop(columns=["snapshot_date", "label", "bbl"], errors="ignore")
        X = align_columns_to_model(X, model)
        # Predict
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)

        st.subheader("Prediction")
        st.write(f"**Building ID (BBL):** {building_id}")
        st.write(f"**Snapshot:** {pd.to_datetime(last)}")
        st.write(f"**Risk Probability:** {prob:.3f}")
        st.write(f"**Predicted Label:** {pred}")

        # 4) SHAP (safe)
        shap_vec = safe_shap_values(model, X)
        with st.expander("Debug: shapes"):
            st.write(f"X columns: {len(X.columns)}")
            st.write(f"X first-row length: {int(X.iloc[0].shape[0])}")
            st.write(f"SHAP vector length: {None if shap_vec is None else len(shap_vec)}")

        if shap_vec is not None:
            fi = build_feature_importance_df(X.iloc[0], shap_vec)
            if not fi.empty:
                st.subheader("Top Features Driving Risk")
                st.dataframe(fi.head(10))
                st.bar_chart(fi.set_index("feature")["shap_value"].head(10))
            else:
                st.info("SHAP produced no comparable features; skipping explanation.")
        else:
            st.info("SHAP explanation not available for this run.")

    except Exception as e:
        st.error(f"💥 Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc())
