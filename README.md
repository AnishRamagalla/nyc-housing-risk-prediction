#  NYC Housing Risk Prediction  

## Project Overview  
This project aims to build a **machine learning model that predicts housing quality risks in New York City**, using publicly available data from NYC Open Data. The model will identify which buildings are most likely to develop serious housing violations (such as mold, heating issues, or rodent infestations).  

Unlike existing dashboards that only show past complaints, this project focuses on:  
-  Predicting future housing risks  
-  Auditing fairness across neighborhoods  
-  Explaining predictions using interpretable AI methods  

---

##  Objectives  
- **Predictive modeling** – Train ML models on 311 complaints and housing violation data.  
- **Fairness Analysis** – Investigate if predictions are biased against certain neighborhoods or demographics.  
- **Interpretability** – Apply SHAP/LIME to explain why a property is flagged as “high-risk.”  

---

## 🧱 Architecture

NYC Open Data (DOB Complaints)
↓ (ingest)
data/live/complaints.sqlite
↓ (feature update)
data/live/features.csv
↓ (predict)
models/monthly_rf_v1.joblib
↓
Streamlit Web App (app.py)


---

## 📦 Project layout

├── app.py # Streamlit app
├── live/
│ └── ingest_complaints.py # pulls recent DOB complaints → SQLite
├── src/
│ ├── features.py # builds rolling features per BIN
│ ├── predict_live.py # CLI prediction on latest features
│ ├── train.py # trains RandomForest, saves .joblib
│ └── ... # (preprocess, evaluate, etc.)
├── models/
│ └── monthly_rf_v1.joblib # trained model artifact
├── data/
│ ├── live/ # runtime data (SQLite, features.csv)
│ └── processed/ # training data samples
├── notebooks/ # 01_ingestion, 02_eda, 03_model
├── requirements.txt
└── .gitignore


---

## 🚀 Quickstart (local)

```bash
# 1) create venv
python -m venv venv
venv\Scripts\activate  # on Windows (or: source venv/bin/activate)

# 2) install deps
pip install -r requirements.txt

# 3) (optional) fetch recent complaints locally and build features
python live/ingest_complaints.py
python src/features.py

# 4) (optional) train model locally
python src/train.py

# 5) run app
streamlit run app.py


🧪 Reproducible modeling

CV: TimeSeriesSplit(n_splits=5)

Metrics: ROC-AUC, PR-AUC; calibration available in reports

Artifacts:

models/monthly_rf_v1.joblib

reports/metrics.json, reports/feature_importance.csv

🗺️ Roadmap

 SHAP & permutation importance in the app (per-prediction explanations)

 Map view with color-coded risk markers (pydeck)

 Auto-refresh features in the cloud (scheduled job / GitHub Actions)

 FastAPI backend (/predict, /features)

 Next.js (React) frontend with search + map + explanations


 🙌 Notes

Building key: BIN (Building Identification Number)

Data: NYC Open Data (DOB Complaints). Used under their terms.

Issues / ideas? PRs welcome → <YOUR-REPO-URL>