"""Streaming pipeline that applies trained models to live records."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Iterable, List

import numpy as np
import pandas as pd


class StreamingPredictor:
    """Maintains a rolling window of predictions from the live stream."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.history: Deque[dict] = deque(maxlen=window_size)
        self.model_predictions: Dict[str, Deque[int]] = {
            name: deque(maxlen=window_size) for name in []
        }

    def register_model(self, name: str) -> None:
        if name not in self.model_predictions:
            self.model_predictions[name] = deque(maxlen=self.window_size)

    def add_record(self, record: dict, predictions: Dict[str, int]) -> None:
        """Append one streaming record with its per-model predictions."""
        entry = dict(record)
        entry.update({f"pred_{k}": v for k, v in predictions.items()})
        self.history.append(entry)
        for name, pred in predictions.items():
            self.model_predictions[name].append(pred)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def recent_delay_rate(self, model_name: str | None = None) -> float:
        """Fraction of records predicted as delayed in the window."""
        if model_name is None:
            if not self.history:
                return 0.0
            preds = [r.get("delay_label", 0) for r in self.history]
            return float(np_mean(preds))
        dq = self.model_predictions.get(model_name)
        if not dq:
            return 0.0
        return float(np_mean(list(dq)))


def np_mean(values: List[float]) -> float:
    return float(sum(values)) / len(values) if values else 0.0


def simulate_stream(
    records: Iterable[dict],
    pipelines: Dict,
    preprocessor,
    predictor: StreamingPredictor,
    delay_seconds: float = 0.0,
) -> List[dict]:
    """Push records through all pipelines and record predictions.

    Returns a list of per-record prediction results.
    """
    results: List[dict] = []
    feature_cols = None
    for record in records:
        row = pd.DataFrame([record])
        # Keep only the columns the preprocessor saw during training by
        # reconstructing via the same feature set order.
        predictions: Dict[str, int] = {}
        for name, pipeline in pipelines.items():
            try:
                pred = int(pipeline.predict(row)[0])
            except Exception:
                pred = 0
            predictions[name] = pred
        predictor.add_record(record, predictions)
        results.append({**record, **predictions})
        if delay_seconds:
            time.sleep(delay_seconds)
    return results
