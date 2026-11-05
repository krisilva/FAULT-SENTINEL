"""
Fault Sentinel — Preprocessing & Prediction Helpers
Feature-column schemas, class-label maps, and the actual predict-from-
input logic for each module. Keeping this separate from the page files
means Detection/Classification/Prediction pages stay focused on layout,
while the "what do we feed the model and how do we read its output"
logic lives in one place.
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Feature schemas (must match each model's training column order)
# ------------------------------------------------------------------ #
DETECTION_FEATURES = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic",
                       "Active_Power_kW", "Reactive_Power_kVAR", "Frequency_Hz"]

CLASSIFICATION_FEATURES = DETECTION_FEATURES  # identical schema

PREDICTION_FEATURES = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic",
                        "Active_Power", "Reactive_Power", "Frequency"]

WINDOW = 10  # Module 3 rolling-window size

CLASS_LABELS = {
    0: "Normal", 1: "Overload", 2: "HRGF", 3: "LL Fault", 4: "LLG Fault",
    5: "LLLG Fault", 6: "SGF/LG Fault", 7: "Overcurrent Conditions",
    8: "Voltage Sag", 9: "Voltage Swell", 10: "Transformer Abnormalities",
}


# ------------------------------------------------------------------ #
# Module 1 — Detection
# ------------------------------------------------------------------ #
def predict_detection_row(model, values: dict):
    """Single reading -> (label, fault_probability)."""
    row = pd.DataFrame([values])[DETECTION_FEATURES]
    proba = float(model.predict_proba(row)[:, 1][0])
    label = "Fault" if proba >= 0.5 else "Normal"
    return label, proba


def predict_detection_df(model, df: pd.DataFrame) -> pd.DataFrame:
    """Batch predict -> df with Predicted_Label + Fault_Probability columns added."""
    X = df[DETECTION_FEATURES]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["Predicted_Label"] = np.where(pred == 1, "Fault", "Normal")
    out["Fault_Probability"] = proba.round(4)
    return out


# ------------------------------------------------------------------ #
# Module 2 — Classification
# ------------------------------------------------------------------ #
def predict_classification_row(model, values: dict):
    """Single reading -> (top_class_name, top_probability, full_probability_table)."""
    row = pd.DataFrame([values])[CLASSIFICATION_FEATURES]
    proba = model.predict_proba(row)[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1]
    top_name = CLASS_LABELS.get(int(classes[order[0]]), str(classes[order[0]]))
    top_prob = float(proba[order[0]])
    table = pd.DataFrame({
        "Fault Type": [CLASS_LABELS.get(int(c), str(c)) for c in classes],
        "Probability": proba,
    }).sort_values("Probability", ascending=False).reset_index(drop=True)
    return top_name, top_prob, table


def predict_classification_df(model, df: pd.DataFrame) -> pd.DataFrame:
    proba_all = model.predict_proba(df[CLASSIFICATION_FEATURES])
    pred_idx = np.argmax(proba_all, axis=1)
    classes = model.classes_
    out = df.copy()
    out["Predicted_Fault_Type"] = [CLASS_LABELS.get(int(classes[i]), str(classes[i])) for i in pred_idx]
    out["Confidence"] = proba_all.max(axis=1).round(4)
    return out


# ------------------------------------------------------------------ #
# Module 3 — Early Prediction (hybrid LSTM + XGBoost soft-voting)
# ------------------------------------------------------------------ #
def hybrid_predict_window(window_df: pd.DataFrame, xgb_model, lstm_model, scaler,
                           label_encoder, lstm_weight: float, xgb_weight: float):
    """
    window_df: WINDOW rows x 9 PREDICTION_FEATURES columns (oldest -> newest).
    Returns (predicted_label, confidence, full_probability_array, class_names).
    """
    flat = window_df[PREDICTION_FEATURES].values.reshape(1, -1)
    flat_scaled = scaler.transform(flat)
    seq_scaled = flat_scaled.reshape(1, WINDOW, len(PREDICTION_FEATURES))

    lstm_proba = lstm_model.predict(seq_scaled, verbose=0)[0]
    xgb_proba = xgb_model.predict_proba(flat_scaled)[0]
    hybrid = lstm_weight * lstm_proba + xgb_weight * xgb_proba

    idx = int(np.argmax(hybrid))
    label = label_encoder.classes_[idx]
    return label, float(hybrid[idx]), hybrid, label_encoder.classes_


def severity(label: str) -> int:
    """Maps a predicted tier label to an urgency level: 0=safe .. 3=critical."""
    if label == "No_Imminent_Fault":
        return 0
    if label.endswith("60min"):
        return 1
    if label.endswith("30min"):
        return 2
    if label.endswith("15min"):
        return 3
    return 0


SEVERITY_TEXT = {
    0: "No Imminent Fault",
    1: "Warning · ~60 min",
    2: "Warning · ~30 min",
    3: "Critical · ~15 min",
}
