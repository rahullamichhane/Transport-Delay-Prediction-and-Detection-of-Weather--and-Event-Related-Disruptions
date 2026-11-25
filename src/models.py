"""Machine learning models for delay classification (4-5 models)."""

from __future__ import annotations

from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

try:  # XGBoost is optional; fall back gracefully if unavailable
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # pragma: no cover - environment dependent
    HAS_XGB = False


MODEL_CONFIGS: Dict[str, BaseEstimator] = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
}

if HAS_XGB:
    MODEL_CONFIGS["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric="logloss",
    )


def build_model_pipeline(name: str, preprocessor) -> Pipeline:
    """Wrap a model with the shared preprocessing pipeline."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_CONFIGS)}")
    model = MODEL_CONFIGS[name]
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["report"] = classification_report(y_true, y_pred, zero_division=0)
    return metrics


def train_and_evaluate(
    name: str,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, dict]:
    """Train one model pipeline and evaluate on the test set."""
    pipeline = build_model_pipeline(name, preprocessor)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob)
    return pipeline, metrics


def train_all_models(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Pipeline], Dict[str, dict]]:
    """Train every configured model and return pipelines + metrics."""
    pipelines: Dict[str, Pipeline] = {}
    results: Dict[str, dict] = {}
    for name in MODEL_CONFIGS:
        pipeline, metrics = train_and_evaluate(
            name, preprocessor, X_train, y_train, X_test, y_test
        )
        pipelines[name] = pipeline
        results[name] = metrics
    return pipelines, results


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)
