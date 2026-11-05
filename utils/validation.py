"""
Fault Sentinel - Input Validation
Drop this file into  utils/validation.py

validate_csv()  -> checks an uploaded CSV before it reaches any model
validate_row()  -> checks manual-entry values (dict) before prediction

Both return (clean_data, errors, warnings):
  errors   -> STOP, show with st.error, do not run the model
  warnings -> run the model but show with st.warning
"""

import difflib
import numpy as np
import pandas as pd

# Column names that are the same thing under a different name
# (Detection/Classification use *_kW / *_Hz, Prediction does not)
ALIASES = {
    "active_power": "Active_Power_kW", "active_power_kw": "Active_Power_kW",
    "reactive_power": "Reactive_Power_kVAR", "reactive_power_kvar": "Reactive_Power_kVAR",
    "frequency": "Frequency_Hz", "frequency_hz": "Frequency_Hz",
}

# (hard_min, hard_max) -> physically impossible outside this  => ERROR
# (soft_min, soft_max) -> outside the training-data range     => WARNING
# Soft ranges come from data/*.csv with a safety margin.
LIMITS = {
    "Va": ((0, 50), (0, 10)), "Vb": ((0, 50), (0, 10)), "Vc": ((0, 50), (0, 10)),
    "Ia": ((0, 100000), (0, 20000)), "Ib": ((0, 100000), (0, 20000)),
    "Ic": ((0, 100000), (0, 20000)),
    "Active_Power_kW": ((0, 1e7), (0, 250000)),
    "Reactive_Power_kVAR": ((-1e7, 1e7), (0, 400000)),
    "Frequency_Hz": ((40, 60), (49.5, 50.5)),
}

LABEL_COLUMNS = ["Label", "Fault_Type", "Predicted_Label"]


def _canon(name: str) -> str:
    """'  active power ' -> 'Active_Power_kW' (also fixes case/spaces)."""
    key = str(name).strip().replace(" ", "_").lower()
    return ALIASES.get(key, None) or str(name).strip()


def _limits_for(col):
    key = _canon(col)
    return LIMITS.get(key) or LIMITS.get({"Active_Power": "Active_Power_kW",
                                           "Reactive_Power": "Reactive_Power_kVAR",
                                           "Frequency": "Frequency_Hz"}.get(col, col))


def _rows(mask, limit=5):
    idx = [str(i + 2) for i in np.flatnonzero(mask)[:limit]]  # +2 = Excel row (header = 1)
    more = "..." if mask.sum() > limit else ""
    return ", ".join(idx) + more


# ---------------------------------------------------------------- #
def validate_labels(series: pd.Series, allowed):
    """Check a label column. `allowed` = list of valid values (names or numbers).
    Returns (cleaned_series, errors)."""
    errors = []
    allowed_str = {str(a).strip().lower(): a for a in allowed}
    cleaned, bad = [], {}
    for v in series:
        key = str(v).strip().lower()
        if key.endswith(".0"):
            key = key[:-2]
        if key in allowed_str:
            cleaned.append(allowed_str[key])
        else:
            cleaned.append(np.nan)
            bad[str(v)] = bad.get(str(v), 0) + 1
    for v, n in list(bad.items())[:5]:
        hint = difflib.get_close_matches(v.lower(), allowed_str.keys(), n=1)
        tip = f" - did you mean '{allowed_str[hint[0]]}'?" if hint else ""
        errors.append(f"Invalid label '{v}' ({n} rows){tip}. Allowed: {', '.join(map(str, allowed))}")
    return pd.Series(cleaned, index=series.index), errors


# ---------------------------------------------------------------- #
def validate_csv(df: pd.DataFrame, features, allowed_labels=None, label_col=None,
                 min_rows=1, require_timestep_order=False):
    errors, warnings = [], []

    if df is None or df.empty:
        return None, ["The file is empty (no data rows)."], warnings
    if len(df) < min_rows:
        return None, [f"Need at least {min_rows} rows, got {len(df)}."], warnings

    # 1. clean + map column names (handles ' Va', 'va', 'Active_Power' ...)
    df = df.copy()
    rename = {}
    for c in df.columns:
        if str(c).strip() in features:
            rename[c] = str(c).strip()
            continue
        lower = {f.lower(): f for f in features}
        if str(c).strip().lower() in lower:
            rename[c] = lower[str(c).strip().lower()]
        elif _canon(c) in features:
            rename[c] = _canon(c)
    df = df.rename(columns=rename)
    if df.columns.duplicated().any():
        errors.append("Duplicate column names: " + ", ".join(df.columns[df.columns.duplicated()]))
        return None, errors, warnings

    # 2. missing columns (+ spelling suggestion)
    missing = [f for f in features if f not in df.columns]
    if missing:
        extra = [c for c in df.columns if c not in features]
        for m in missing:
            hint = difflib.get_close_matches(m, extra, n=1, cutoff=0.6)
            errors.append(f"Missing column '{m}'" + (f" - found '{hint[0]}', is that a typo?" if hint else ""))
        errors.append("This does not look like the right file for this module. "
                      f"Expected columns: {', '.join(features)}")
        return None, errors, warnings

    # 3. numeric check - text in a number column
    _text_bad = {}
    for c in features:
        num = pd.to_numeric(df[c], errors="coerce")
        text_bad = num.isna() & df[c].notna() & (df[c].astype(str).str.strip() != "")
        if text_bad.any():
            errors.append(f"Column '{c}' has non-numeric values (e.g. '{df.loc[text_bad, c].iloc[0]}') "
                          f"in rows: {_rows(text_bad.values)}")
        df[c] = num
        _text_bad[c] = text_bad.values

    # 4. blanks / inf (rows already reported as text are skipped)
    for c in features:
        blank = (df[c].isna() | np.isinf(df[c])) & ~_text_bad[c]
        if blank.any():
            errors.append(f"Column '{c}' has {int(blank.sum())} blank/infinite values, rows: {_rows(blank.values)}")

    # 5. range checks (only on valid numbers)
    for c in features:
        lim = _limits_for(c)
        if not lim:
            continue
        (hmin, hmax), (smin, smax) = lim
        col = df[c]
        hard = ((col < hmin) | (col > hmax)).fillna(False)
        if hard.any():
            errors.append(f"Column '{c}' has impossible values ({hmin} to {hmax} allowed), rows: {_rows(hard.values)}")
        soft = (((col < smin) | (col > smax)) & ~hard).fillna(False)
        if soft.any():
            warnings.append(f"Column '{c}': {int(soft.sum())} values are outside the range the model was "
                            f"trained on ({smin} - {smax}); results may be less reliable.")

    # 6. label column (optional)
    if label_col and label_col in df.columns and allowed_labels is not None:
        df[label_col], lab_err = validate_labels(df[label_col], allowed_labels)
        errors += lab_err

    # 7. time-series checks (prediction module)
    if require_timestep_order and "Timestep" in df.columns:
        ts = pd.to_numeric(df["Timestep"], errors="coerce")
        if ts.isna().any():
            errors.append("Column 'Timestep' has non-numeric values.")
        elif ts.duplicated().any():
            warnings.append("Duplicate Timestep values found - check the file is a single episode.")
        elif not ts.is_monotonic_increasing:
            warnings.append("Timestep is not in order - rows will be sorted by Timestep.")

    if errors:
        return None, errors, warnings
    return df, errors, warnings


# ---------------------------------------------------------------- #
def validate_row(values: dict, features):
    """Manual entry. Works for st.number_input dict and for single data_editor rows."""
    errors, warnings = [], []
    clean = {}
    for f in features:
        raw = values.get(f)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{f}' must be a number (got '{raw}').")
            continue
        if np.isnan(v) or np.isinf(v):
            errors.append(f"'{f}' is empty or invalid.")
            continue
        clean[f] = v
        lim = _limits_for(f)
        if lim:
            (hmin, hmax), (smin, smax) = lim
            if v < hmin or v > hmax:
                errors.append(f"'{f}' = {v} is physically impossible (allowed {hmin} to {hmax}).")
            elif v < smin or v > smax:
                warnings.append(f"'{f}' = {v} is outside the training range ({smin} - {smax}).")
    return (None if errors else clean), errors, warnings


def show(st, errors, warnings):
    """Helper: render messages in Streamlit. Returns True if it is safe to continue."""
    for w in warnings:
        st.warning(w)
    if errors:
        st.error("Please fix the following before running the model:")
        for e in errors:
            st.markdown(f"- {e}")
        return False
    return True
