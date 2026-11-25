"""Preprocessing utilities for the transport delay dataset."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "is_peak_hour",
    "is_weekend",
    "temperature",
    "humidity",
]
CATEGORICAL_FEATURES = ["vehicle_class", "route_type", "weather_condition", "event_type"]
TARGET = "delay_label"
DROP_FEATURES = ["record_id", "timestamp", "vehicle_id", "route_id", "stop_id", "delay_minutes"]


def build_preprocessor() -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features(df: pd.DataFrame, target: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix X and target vector y."""
    X = df.drop(columns=[c for c in DROP_FEATURES + [target] if c in df.columns])
    y = df[target]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
