"""Shared feature engineering and inference helpers for career prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "india_youth_career_choices.csv"
MODEL_DIR = ROOT_DIR / "models"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
MODEL_PATH = MODEL_DIR / "career_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

SKILL_COLUMNS = [
    "Programming_Skill_1_10",
    "Communication_Skill_1_10",
    "Analytical_Skill_1_10",
    "Creative_Design_Skill_1_10",
]
NUMERIC_FEATURES = ["Age", "Academic_Score", *SKILL_COLUMNS]
CATEGORICAL_FEATURES = [
    "Gender",
    "Region",
    "City_Tier",
    "Stream_12th",
    "Current_Degree",
    "Primary_Interest",
    "Influencing_Factor",
]
FEATURE_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
TARGET_COLUMN = "Preferred_Career"


def cgpa_to_percentage(cgpa: Any, cgpa_fit: dict[str, float]) -> Any:
    """Convert CGPA values using the linear fit saved with the trained model."""
    slope = float(cgpa_fit["slope"])
    intercept = float(cgpa_fit["intercept"])
    converted = np.asarray(cgpa, dtype=float) * slope + intercept
    converted = np.clip(converted, 0.0, 100.0)
    if converted.ndim == 0:
        return float(converted)
    return converted


def engineer_features(df: pd.DataFrame, cgpa_fit: dict[str, float]) -> pd.DataFrame:
    """Build the shared model feature frame from raw student profile columns."""
    features = df.copy()
    for column in ["Age", "Academic_Percentage", "Current_CGPA", *SKILL_COLUMNS]:
        if column not in features:
            features[column] = np.nan
        features[column] = pd.to_numeric(features[column], errors="coerce")

    academic_score = features["Academic_Percentage"].copy()
    missing_score = academic_score.isna()
    if missing_score.any():
        academic_score.loc[missing_score] = cgpa_to_percentage(
            features.loc[missing_score, "Current_CGPA"].to_numpy(), cgpa_fit
        )
    features["Academic_Score"] = academic_score.clip(lower=0.0, upper=100.0)

    for column in CATEGORICAL_FEATURES:
        if column not in features:
            features[column] = np.nan

    return features.loc[:, FEATURE_COLUMNS].copy()


def load_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    """Load the fitted preprocessor, classifier, and metadata from disk."""
    import json

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    return preprocessor, model, metadata


def predict_profile(
    profile_dict: dict[str, Any],
    preprocessor: Any,
    model: Any,
    metadata: dict[str, Any],
) -> pd.Series:
    """Return model probability estimates for one profile, highest first."""
    if not metadata.get("supports_proba", False) or not hasattr(model, "predict_proba"):
        raise ValueError("The saved model does not support predict_proba.")

    profile = pd.DataFrame([profile_dict])
    features = engineer_features(profile, metadata["cgpa_fit"])
    transformed = preprocessor.transform(features)
    probabilities = model.predict_proba(transformed)[0]
    class_names = list(model.classes_)
    if len(probabilities) != len(class_names):
        raise ValueError("Model class labels do not match its probability output.")
    return pd.Series(probabilities, index=class_names, name="probability").sort_values(
        ascending=False
    )
