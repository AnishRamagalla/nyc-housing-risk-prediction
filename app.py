# app.py
import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config("NYC Housing Risk Prediction", layout="wide")

MODEL_PATH = "artifacts/model.pkl"
FEATURE_LIST_PATH = "artifacts/feature_list.json"
LIVE_FEATURES_PATH = "data/live/features.csv"

@st.cache_data
def load_model_and_features():
    # Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Feature list
    if not os.path.exists(FEATURE_LIST_PATH):
        raise FileNotFoundError(f"Missing feature_list.json: {FEATURE_LIST_PATH}")
    with open(FEATURE_LIST_PATH) as f:
        feature_list = json.load(f)

    # Live features
    if not os.path.exists(LIVE_FEATURES_PATH):
        raise FileNotFoundError(f"Missing live features: {LIVE_FEATURES_PATH}")
    df = pd.read_csv(LIVE_FEATURES_PATH)

    # Ensure columns exist; fill if missing
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    return model, feature_list, df

def predict_for_bbl(bbl: str, model, feature_list, df_features):
    row = df_features[df_features["bbl"].astype(str) == str(bbl)]
    if row.empty:
        return None, f"No data found for building ID {bbl}"
    X = row[feature_list].fillna(0)
    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= 0.5)
    return (pred, prob), None

def main():
    st.title("🏠 NYC Housing Risk Prediction")

    try:
        model, feature_list, df_features = load_model_and_features()
    except Exception as e:
        st.error(f"Startup error: {e}")
        st.stop()

    bbl = st.text_input("Enter a BBL (e.g. 1004500035)", value="1004500035")
    if st.button("Predict"):
        out, err = predict_for_bbl(bbl, model, feature_list, df_features)
        if err:
            st.warning(err)
        else:
            pred, prob = out
            st.success(f"Prediction={pred}, Prob={prob:.3f}")

    with st.expander("Debug"):
        st.write("rows:", len(df_features))
        st.write("first rows:", df_features.head())
        st.write("features used:", feature_list)

if __name__ == "__main__":
    main()
