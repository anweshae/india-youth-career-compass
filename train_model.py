"""Train and evaluate the career classifier separately from the Streamlit app."""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, top_k_accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.preprocessing import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    engineer_features,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def fit_cgpa_relationship(training_data: pd.DataFrame) -> dict[str, float]:
    """Fit the CGPA-to-percentage linear relationship on training rows only."""
    points = training_data.loc[:, ["Current_CGPA", "Academic_Percentage"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(points) < 2:
        raise ValueError("At least two rows with both CGPA and percentage are required.")
    slope, intercept = np.polyfit(points["Current_CGPA"], points["Academic_Percentage"], 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def build_preprocessor() -> ColumnTransformer:
    """Create the required numeric and categorical preprocessing steps."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_candidates() -> dict[str, object]:
    """Return the three classifier candidates specified in the project brief."""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=8,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def main() -> None:
    """Select a classifier by CV, evaluate held-out data, and save artifacts."""
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data:
        raise ValueError(f"Required target column {TARGET_COLUMN!r} is missing.")
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    y = data[TARGET_COLUMN].astype(str)
    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    training_data = data.iloc[train_indices]
    y_train = y.iloc[train_indices]
    test_data = data.iloc[test_indices]
    y_test = y.iloc[test_indices]

    cgpa_fit = fit_cgpa_relationship(training_data)
    x_train = engineer_features(training_data, cgpa_fit)
    x_test = engineer_features(test_data, cgpa_fit)

    candidates = build_candidates()
    cv_scores: dict[str, tuple[float, float]] = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)]
        )
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=CV_FOLDS,
            scoring="f1_macro",
            n_jobs=1,
        )
        cv_scores[name] = (float(scores.mean()), float(scores.std()))
        print(f"{name} CV macro-F1: {scores.mean():.4f} +/- {scores.std():.4f}")

    winner_name = max(cv_scores, key=lambda name: cv_scores[name][0])
    winner = candidates[winner_name]
    preprocessor = build_preprocessor()
    transformed_train = preprocessor.fit_transform(x_train, y_train)
    winner.fit(transformed_train, y_train)
    transformed_test = preprocessor.transform(x_test)

    if not hasattr(winner, "predict_proba"):
        raise TypeError(f"Selected classifier {winner_name} does not provide predict_proba.")
    predicted = winner.predict(transformed_test)
    probabilities = winner.predict_proba(transformed_test)
    accuracy = float(accuracy_score(y_test, predicted))
    top3_accuracy = float(
        top_k_accuracy_score(y_test, probabilities, k=3, labels=winner.classes_)
    )
    macro_f1 = float(f1_score(y_test, predicted, average="macro", zero_division=0))
    majority_class = y_train.value_counts().idxmax()
    majority_baseline = float((y_test == majority_class).mean())
    class_recalls = recall_score(
        y_test,
        predicted,
        labels=winner.classes_,
        average=None,
        zero_division=0,
    )
    per_class_recall = {
        str(label): float(recall)
        for label, recall in zip(winner.classes_, class_recalls)
    }
    metrics = {
        "accuracy": accuracy,
        "top_3_accuracy": top3_accuracy,
        "macro_f1": macro_f1,
        "majority_class_baseline": majority_baseline,
        "majority_class": str(majority_class),
    }
    metadata = {
        "model_name": winner_name,
        "classes": [str(label) for label in winner.classes_],
        "metrics": metrics,
        "per_class_recall": per_class_recall,
        "cgpa_fit": cgpa_fit,
        "supports_proba": True,
        "data_note": "Training data is synthetic; metrics describe this synthetic dataset only.",
        "feature_columns": [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
        "cv_macro_f1": {
            name: {"mean": mean, "std": std}
            for name, (mean, std) in cv_scores.items()
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH, compress=3)
    joblib.dump(winner, MODEL_PATH, compress=3)
    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"\nSelected model: {winner_name}")
    print(f"Holdout accuracy: {accuracy:.4f}")
    print(f"Holdout top-3 accuracy: {top3_accuracy:.4f}")
    print(f"Holdout macro-F1: {macro_f1:.4f}")
    print(f"Majority-class baseline ({majority_class}): {majority_baseline:.4f}")
    print("Per-class recall:")
    for label, recall in per_class_recall.items():
        print(f"  {label}: {recall:.4f}")
    print(
        "Data Scientist and Software Engineer are nearly indistinguishable in this "
        "synthetic data; this overlap is reported as observed, not tuned away."
    )
    for artifact_path in (PREPROCESSOR_PATH, MODEL_PATH, METADATA_PATH):
        print(f"Saved {artifact_path.relative_to(MODEL_DIR.parent)} ({artifact_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

