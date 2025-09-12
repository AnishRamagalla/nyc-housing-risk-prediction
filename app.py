import streamlit as st
import pandas as pd
import joblib
import os

# --- Config ---
MODEL_PATH = "models/monthly_rf_v1.joblib"
FEATURES_FILE = "data/live/features.csv"
# Optional: add building coordinates here or load from CSV
BUILDING_COORDS = {
    "1012607": {"lat": 40.741, "lon": -74.003},  # Example BIN -> Chelsea, Manhattan
    "1085410": {"lat": 40.678, "lon": -73.944},  # Example BIN -> Brooklyn
    # Add more BINs as needed
}

# --- Load model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run training first.")
        return None
    return joblib.load(MODEL_PATH)

# --- Predict ---
def align_columns(df, model):
    need = list(model.feature_names_in_)
    for c in need:
        if c not in df.columns:
            df[c] = 0.0
    extra = [c for c in df.columns if c not in need]
    if extra:
        df = df.drop(columns=extra)
    return df[need]

def predict(building_id, model):
    if not os.path.exists(FEATURES_FILE):
        st.error("Features file not found. Run src/features.py first.")
        return None
    df = pd.read_csv(FEATURES_FILE, parse_dates=["snapshot_date"])
    row = df[df["building_id"].astype(str) == str(building_id)]
    if row.empty:
        st.warning(f"No features found for building {building_id}")
        return None
    X = row.drop(columns=["building_id", "snapshot_date"], errors="ignore")
    X = align_columns(X, model)
    prob = model.predict_proba(X)[0,1]
    return {
        "building_id": building_id,
        "snapshot": str(row["snapshot_date"].iloc[0]),
        "risk_prob": float(prob),
        "pred_label": int(prob >= 0.5),
    }

# --- Streamlit UI ---
st.set_page_config(page_title="NYC Housing Risk", layout="wide")
st.title("🏠 NYC Housing Risk Prediction")
st.write("Enter a Building ID (BIN) to see its latest complaint-based risk score.")

model = load_model()

bid = st.text_input("Building ID (BIN)", "1012607")

if st.button("Predict Risk"):
    if model:
        result = predict(bid, model)
        if result:
            st.subheader("Prediction Result")
            st.write(f"**Building ID:** {result['building_id']}")
            st.write(f"**Snapshot Date:** {result['snapshot']}")
            st.write(f"**Risk Probability:** {result['risk_prob']:.3f}")
            st.write(f"**Predicted Label:** {result['pred_label']}")

            # Show on map if coords available
            if str(bid) in BUILDING_COORDS:
                coords = BUILDING_COORDS[str(bid)]
                st.map(pd.DataFrame([{"lat": coords["lat"], "lon": coords["lon"]}]))
            else:
                st.info("Coordinates not available for this BIN. Add to BUILDING_COORDS.")
