"""
Module: models.py
Purpose: Train and cache all ML models used across the application.
Models:
  - Logistic Regression  → Health Prediction (binary: Healthy / Unhealthy)
  - Random Forest        → Disease Classification (multi-class Disease_Label)
  - Linear Regression    → Yield Prediction (continuous Yield)
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model    import LogisticRegression, LinearRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
# (No metrics imported as comparison page was removed)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering Helper
# ─────────────────────────────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
    "Temperature", "Humidity", "Rainfall", "pH_Level",
    "Nitrogen", "Phosphorus", "Potassium", "Leaf_Spots",
    "Irrigation_Level", "Sunlight_Hours", "Soil_Moisture",
    "NDVI_Index", "Leaf_Area", "Chlorophyll_Level", "Yield",
]

CATEGORICAL_FEATURES = [
    "Crop_Type", "Soil_Type", "Leaf_Color", "Growth_Stage",
    "Region", "Season", "Pest_Presence", "Fertilizer_Used",
    "Water_Quality", "Air_Quality", "Stress_Level", "Disease_Severity",
]


def _encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    One-hot encode categorical features and return the encoded DataFrame
    plus a dict of fitted LabelEncoders for later inference.
    """
    le_dict = {}
    df_enc  = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            le_dict[col] = le
    return df_enc, le_dict


def _prepare_X(df_enc: pd.DataFrame, include_yield: bool = False) -> pd.DataFrame:
    """
    Select feature columns used for training (exclude targets).

    Parameters
    ----------
    include_yield : bool
        If True, Yield is kept as a feature (used only for the Linear Regression
        yield predictor where Yield is the TARGET and must not appear as a feature).
        NOTE: For yield prediction the target IS Yield, so this parameter is
        actually unused in yield training — Yield is always excluded as a feature.
    """
    # Always exclude these columns from the feature matrix
    exclude = {"Disease_Label", "Healthy", "Disease_Type", "Yield"}
    feature_cols = [c for c in df_enc.columns if c not in exclude]
    return df_enc[feature_cols]


# ─────────────────────────────────────────────────────────────────────────────
# Cached Model Training (runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🌿 Training ML models on agricultural data…")
def train_all_models(df: pd.DataFrame):
    """
    Train all three models and return a dictionary with models, encoders,
    scalers, and evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame – The full agricultural dataset.

    Returns
    -------
    dict with keys:
        lr_model, rf_model, linreg_model,
        le_dict, le_disease, scaler,
        lr_metrics, rf_metrics, linreg_metrics,
        feature_cols
    """
    # ── Encode categorical features ─────────────────────────────────────────
    df_enc, le_dict = _encode_features(df)

    # ── Encode target for Disease Classification ─────────────────────────────
    le_disease = LabelEncoder()
    y_disease  = le_disease.fit_transform(df_enc["Disease_Label"].astype(str))

    X = _prepare_X(df_enc)
    feature_cols = list(X.columns)

    # ── Scale features ───────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train/Test Split ─────────────────────────────────────────────────────
    X_tr, X_te, yh_tr, yh_te = train_test_split(
        X_scaled, df_enc["Healthy"].values, test_size=0.2, random_state=42)

    _, _, yd_tr, yd_te = train_test_split(
        X_scaled, y_disease, test_size=0.2, random_state=42)

    _, _, yy_tr, yy_te = train_test_split(
        X_scaled, df_enc["Yield"].values, test_size=0.2, random_state=42)

    # ── 1. Logistic Regression – Health Prediction ───────────────────────────
    lr_model = LogisticRegression(max_iter=500, random_state=42)
    lr_model.fit(X_tr, yh_tr)

    # ── 2. Random Forest – Disease Classification ────────────────────────────
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_tr, yd_tr)

    # ── 3. Linear Regression – Yield Prediction ──────────────────────────────
    linreg_model = LinearRegression()
    linreg_model.fit(X_tr, yy_tr)

    return {
        # Models
        "lr_model"     : lr_model,
        "rf_model"     : rf_model,
        "linreg_model" : linreg_model,
        # Encoders & Scaler
        "le_dict"      : le_dict,
        "le_disease"   : le_disease,
        "scaler"       : scaler,
        "feature_cols" : feature_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-Row Inference Helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_health(models: dict, input_df: pd.DataFrame) -> tuple[str, float]:
    """
    Predict whether a crop is Healthy or Unhealthy.

    Parameters
    ----------
    models   : dict returned by train_all_models()
    input_df : pd.DataFrame with exactly the required feature columns

    Returns
    -------
    (label: str, confidence: float)
    """
    X = _prepare_inference(models, input_df)
    pred  = models["lr_model"].predict(X)[0]
    proba = models["lr_model"].predict_proba(X)[0]
    label = "Healthy ✅" if pred == 1 else "Unhealthy ⚠️"
    conf  = float(proba[pred])
    return label, conf


def predict_disease(models: dict, input_df: pd.DataFrame) -> tuple[str, float]:
    """Predict disease type using Random Forest."""
    X = _prepare_inference(models, input_df)
    pred  = models["rf_model"].predict(X)[0]
    proba = models["rf_model"].predict_proba(X)[0]
    label = models["le_disease"].inverse_transform([pred])[0]
    conf  = float(proba[pred])
    return label, conf


def predict_yield(models: dict, input_df: pd.DataFrame) -> float:
    """Predict crop yield (tonnes/ha) using Linear Regression."""
    X = _prepare_inference(models, input_df)
    yhat = models["linreg_model"].predict(X)[0]
    return max(float(yhat), 0.0)


def _prepare_inference(models: dict, input_df: pd.DataFrame) -> np.ndarray:
    """
    Encode + scale a single-row DataFrame for model inference.
    Fills any missing feature columns with 0.
    Drops Yield and any target columns if present in input.
    """
    df = input_df.copy()
    le_dict = models["le_dict"]

    # Drop target / leakage columns if caller accidentally passed them
    for col in ("Disease_Label", "Healthy", "Disease_Type", "Yield"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode categorical columns
    for col, le in le_dict.items():
        if col in df.columns:
            val = df[col].astype(str).values[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = 0  # unknown category fallback

    # Align columns with training feature set
    for col in models["feature_cols"]:
        if col not in df.columns:
            df[col] = 0
    df = df[models["feature_cols"]]

    return models["scaler"].transform(df)
