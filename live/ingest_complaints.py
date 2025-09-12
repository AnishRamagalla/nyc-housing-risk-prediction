import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ---------------- PATHS ----------------
ARTIFACTS_DIR = "artifacts"
LIVE_FEATURES = "data/live/features.csv"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
FEATURE_LIST_PATH = os.path.join(ARTIFACTS_DIR, "feature_list.json")

# ---------------- LOAD MODEL + FEATURES ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    if not os.path.exists(LIVE_FEATURES):
        return pd.DataFrame()
    return pd.read_csv(LIVE_FEATURES)

@st.cache_data
def load_feature_list():
    import json
    if not os.path.exists(FEATURE_LIST_PATH):
        return []
    with open(FEATURE_LIST_PATH, "r") as f:
        return json.load(f)

# ---------------- PREDICT ----------------
def predict(bbl, model, feature_list, df_features):
    row = df_features[df_features["bbl"].astype(str) == str(bbl)]
    if row.empty:
        return None, f"No data found for building ID {bbl}"

    X = row[feature_list].fillna(0)
    prob = model.predict_proba(X)[0, 1]
    y_pred = model.predict(X)[0]
    return (y_pred, prob), None

# ---------------- UI ----------------
def main():
    st.set_page_config(page_title="NYC Housing Risk Prediction", layout="centered")
    st.title("🏙️ NYC Housing Risk Prediction")

    # Load model + features
    model = load_model()
    df_features = load_features()
    feature_list = load_feature_list()

    if model is None or df_features.empty or not feature_list:
        st.error("❌ Model or live features not available. Please refresh pipeline.")
        return

    # Sidebar helper
    st.sidebar.header("🔎 Explore Available BBLs")
    sample_bbls = df_features["bbl"].astype(str).sample(min(5, len(df_features)), random_state=42).tolist()
    st.sidebar.write("Here are some BBLs you can try:")
    for b in sample_bbls:
        st.sidebar.code(b)

    # Main input
    bbl_input = st.text_input("Enter a BBL (e.g. 1004500035)")
    if st.button("🔮 Predict"):
        if not bbl_input.strip():
            st.warning("Please enter a valid BBL.")
        else:
            result, error = predict(bbl_input, model, feature_list, df_features)
            if error:
                st.warning(error)
            else:
                y_pred, prob = result
                st.success(f"✅ Prediction={y_pred}, Prob={prob:.3f}")

    # Debug panel
    with st.expander("🐞 Debug"):
        st.write("rows:", len(df_features))
        st.dataframe(df_features.head())

    # Freshness info
    if "snapshot_date" in df_features.columns:
        last_update = pd.to_datetime(df_features["snapshot_date"]).max()
        st.caption(f"📅 Data last updated: {last_update.date()}")

if __name__ == "__main__":
    main()
