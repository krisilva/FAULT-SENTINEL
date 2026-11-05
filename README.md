# ⚡ Fault Sentinel

Streamlit dashboard for **fault detection, classification and early fault prediction** in electrical distribution networks.
Undergraduate research project, NSBM Green University, Sri Lanka. Supervisor: Dr. Pabudi Abeyrathne.

| Module | Model | Task |
|---|---|---|
| 1. Detection | XGBoost | Binary: Normal vs Fault |
| 2. Classification | Random Forest | 11-class fault type |
| 3. Early Prediction | LSTM + XGBoost (soft voting) | 10-class warning tier (60/30/15 min) |

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Use Python 3.11 or 3.12 (TensorFlow requirement).

## Admin page
The Admin page is login-protected. On first run the admin account is created with the password from the `ADMIN_PASSWORD` environment variable / Streamlit secret (local fallback: `admin123`). Change it after first login.

## Data availability
The full training datasets (OpenDSS-simulated IEEE 13-bus feeder) are not included. `data/sample/detection_sample.csv` (100 rows) is used by the Overview page. For full data, email **[your-email]** with your name, affiliation and intended use.

## Repo layout
`app.py` + `pages/` (dashboard) · `utils/` (loaders, auth, preprocessing) · `models/` (trained models) · `assets/` (metrics, figures)
