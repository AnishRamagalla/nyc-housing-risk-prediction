# app.py
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium

# ---------------- PATHS ----------------
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURE_LIST_PATH = ARTIFACTS_DIR / "feature_list.json"
LIVE_FEATURES_PATH = Path("data/live/features.csv")

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="NYC Housing Risk Prediction", page_icon="🏠", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

with st.sidebar:
    st.header("📌 About this Project")
    st.write(
        """
        Predict **NYC housing risk** from building complaints.  

        **Features**:
        - Enter or select a BBL  
        - Get risk label & probability  
        - Visualize location on NYC map  
        - Debug panel for dataset  

        👉 Data updates dynamically from live ingestion scripts.
        """
    )
    st.caption("Built with Streamlit + scikit-learn + Folium")

# ---------------- HELPERS ----------------
@st.cache_data
def load_model():
    if not MODEL_PATH.exists():
        return None, "Model file not found", None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return None, f"Failed to load model: {e}", None

    feature_list = None
    if FEATURE_LIST_PATH.exists():
        try:
            with open(FEATURE_LIST_PATH, "r") as f:
                feature_list = json.load(f)
        except Exception as e:
            return model, f"Failed to load feature_list.json: {e}", None

    return model, None, feature_list


@st.cache_data
def load_live_features() -> pd.DataFrame:
    if not LIVE_FEATURES_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LIVE_FEATURES_PATH, dtype={"bbl": "string"})
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        df["bbl"] = df["bbl"].astype("string")
        return df
    except Exception as e:
        st.error(f"Failed to read {LIVE_FEATURES_PATH}: {e}")
        return pd.DataFrame()


def ensure_columns(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    if feature_list is None:
        return df
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    return df


def predict_for_bbl(bbl: str, model, feature_list: list[str], df_features: pd.DataFrame):
    if model is None or df_features.empty:
        return None, None, None

    key = str(bbl).strip()
    row = df_features[df_features["bbl"] == key]

    if row.empty:
        return None, None, None

    X = row[feature_list].fillna(0) if feature_list else row.select_dtypes(include=[np.number]).fillna(0)

    try:
        y_prob = float(model.predict_proba(X)[0, 1])
        y_pred = int(model.predict(X)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, None

    return y_pred, y_prob, row


# ---------------- LOAD DATA ----------------
model, model_err, feature_list = load_model()
df_features = load_live_features()
df_features = ensure_columns(df_features, feature_list)

# ---------------- STATUS ----------------
with st.expander("ℹ️ Status", expanded=False):
    st.write(f"**Model:** {'✅' if MODEL_PATH.exists() else '❌'}")
    st.write(f"**Feature list:** {'✅' if FEATURE_LIST_PATH.exists() else '❌'}")
    st.write(f"**Live features:** {'✅' if LIVE_FEATURES_PATH.exists() else '❌'}")

if model_err:
    st.warning(model_err)

# ---------------- INPUT ----------------
bbl_options = df_features["bbl"].astype(str).unique().tolist() if not df_features.empty else []
default_bbl = bbl_options[0] if bbl_options else ""

col1, col2 = st.columns([3, 1])
with col1:
    bbl_input = st.text_input("Enter or select a BBL (e.g. 1004500035):", value=default_bbl)
with col2:
    picked = st.selectbox("Or pick", options=bbl_options, index=0 if bbl_options else None)
    if picked:
        bbl_input = picked

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Risk", type="primary"):
    if not bbl_input:
        st.warning("Please enter a BBL.")
    elif model is None:
        st.error("Model not loaded. Make sure artifacts/model.pkl exists.")
    elif df_features.empty:
        st.error("No live features loaded. Run the live ingestion job.")
    else:
        y_pred, y_prob, row = predict_for_bbl(bbl_input, model, feature_list, df_features)
        if y_pred is None:
            st.warning(f"No data found for building ID **{bbl_input}**")
        else:
            label = "🟢 Low Risk" if y_pred == 0 else "🔴 High Risk"
            color = "#bbf7d0" if y_pred == 0 else "#fecaca"

            # Risk card
            st.markdown(
                f"""
                <div style="padding:1rem; border-radius:0.5rem; background-color:{color}; border:1px solid #d1d5db">
                    <h3 style="margin:0;">{label}</h3>
                    <p style="margin:0;">Probability of risk: <b>{y_prob:.2%}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Map visualization if lat/lon exists
            if {"lat", "lon"}.issubset(row.columns):
                lat, lon = float(row["lat"].iloc[0]), float(row["lon"].iloc[0])
                m = folium.Map(location=[lat, lon], zoom_start=16)
                folium.Marker([lat, lon], popup=f"BBL {bbl_input} — {label}").add_to(m)
                st_folium(m, width=700, height=450)
            else:
                st.info("⚠️ No latitude/longitude found in live data for this BBL.")

# ---------------- DEBUG ----------------
with st.expander("🛠️ Debug", expanded=False):
    if df_features.empty:
        st.info("No live features loaded.")
    else:
        st.write("rows:", len(df_features))
        st.dataframe(df_features.head(10), use_container_width=True, height=350)
