"""
Fault Sentinel — Model & Data Loaders
All @st.cache_resource / @st.cache_data loaders live here so every page
shares the same cached objects instead of re-loading models from disk.
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

from utils.lstm_architecture import build_lstm_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner=False)
def load_detection_model():
    return joblib.load(os.path.join(MODELS_DIR, "xgboost_detection_model.pkl"))


@st.cache_resource(show_spinner=False)
def load_classification_model():
    return joblib.load(os.path.join(MODELS_DIR, "random_forest_classification_model.pkl"))


@st.cache_resource(show_spinner=False)
def load_prediction_bundle():
    """Returns (xgb_model, lstm_model, scaler, label_encoder)."""
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_prediction_model.pkl"))

    lstm_model = build_lstm_model()
    lstm_model.load_weights(os.path.join(MODELS_DIR, "lstm_prediction_model.weights.h5"))

    scaler = joblib.load(os.path.join(MODELS_DIR, "prediction_scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "prediction_label_encoder.pkl"))
    return xgb_model, lstm_model, scaler, label_encoder


# ------------------------------------------------------------------ #
# Reference datasets (used for the Overview EDA charts & demo samples)
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner=False)
def load_detection_data():
    # Only a 100-row sample ships with the repo (full dataset is not public).
    return pd.read_csv(os.path.join(DATA_DIR, "sample", "detection_sample.csv"))


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner=False)
def load_detection_metrics():
    df = pd.read_csv(os.path.join(ASSETS_DIR, "xgboost_detection_metrics.csv"))
    return df.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def load_classification_metrics():
    with open(os.path.join(ASSETS_DIR, "classification_metrics.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_prediction_metrics():
    with open(os.path.join(ASSETS_DIR, "prediction_metrics.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_module3_comparison():
    return pd.read_csv(os.path.join(ASSETS_DIR, "module3_model_comparison.csv"), index_col=0)
