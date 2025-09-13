import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------- PATHS ----------------
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURE_LIST_PATH = ARTIFACTS_DIR / "feature_list.json"
LIVE_FEATURES_PATH = Path("data/live/features.csv")

# ---------------- UI ----------------
st.set_page_config(page_title="NYC Housing Risk Prediction", page_icon="🏠", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")

# ---------------- LOADERS ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, []
    model = joblib.load(MODEL_PATH)
    if FEATURE_LIST_PATH.exists():
        with open(FEATURE_LIST_PATH, "r") as f:
            feature_list = json.load(f)
    else:
        feature_list = []
    return model, feature_list

@st.cache_data
def load_features():
    if not LIVE_FEATURES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LIVE_FEATURES_PATH, dtype={"bbl": "string"})

# ---------------- PREDICT ----------------
def predict(bbl, model, feature_list, df):
    row = df[df["bbl"].astype(str) == str(bbl)]
    if row.empty:
        return None, None, f"No data found for building ID {bbl}"

    X = row[feature_list].fillna(0)
    try:
        if len(model.classes_) == 1:
            # Only one class in training → probability always 0.0
            y_pred = model.classes_[0]
            y_prob = 0.0
        else:
            y_prob = float(model.predict_proba(X)[0, 1])
            y_pred = int(model.predict(X)[0])
    except Exception as e:
        return None, None, f"Prediction failed: {e}"

    return y_pred, y_prob, None

# ---------------- MAIN ----------------
def main():
    model, feature_list = load_model()
    df = load_features()

    if model is None or df.empty or not feature_list:
        st.error("❌ Model or live features not available. Run the training notebook first.")
        return

    bbl_input = st.text_input("Enter or select a BBL (e.g. 1004500035):")
    options = df["bbl"].astype(str).unique().tolist()
    picked = st.selectbox("Or pick", options=options, index=0 if options else None)
    if picked:
        bbl_input = picked

    if st.button("🔍 Predict Risk", type="primary"):
        y_pred, y_prob, error = predict(bbl_input, model, feature_list, df)
        if error:
            st.error(error)
        else:
            label = "🟢 Low Risk" if y_pred == 0 else "🔴 High Risk"
            st.success(f"{label} — Probability of risk: {y_prob:.2%}")

    # Debug panel
    with st.expander("🛠️ Debug"):
        st.write("rows:", len(df))
        st.dataframe(df.head(10))

if __name__ == "__main__":
    main()
