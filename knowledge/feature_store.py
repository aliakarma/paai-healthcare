"""
feature_store.py
================
Rolling feature cache for the RL policy state construction.
Stores recent vital sign windows and pre-computes trend features.
"""
import numpy as np
from collections import deque


class FeatureStore:
    """Maintains a rolling window of recent vital readings per patient."""

    CHANNELS = ["sbp", "dbp", "glucose_mgdl", "heart_rate", "spo2"]
    MAX_WINDOW = 720  # 60 minutes at 5-min resolution = 12 steps; keep 720 (3 days)

    def __init__(self):
        self._windows: dict[int, dict[str, deque]] = {}

    def push(self, patient_id: int, vital: dict):
        """Add a new vital reading for a patient."""
        if patient_id not in self._windows:
            self._windows[patient_id] = {
                ch: deque(maxlen=self.MAX_WINDOW) for ch in self.CHANNELS}
        for ch in self.CHANNELS:
            self._windows[patient_id][ch].append(vital.get(ch, np.nan))

    def get_rolling_mean(self, patient_id: int, channel: str,
                          window_steps: int = 12) -> float:
        if patient_id not in self._windows:
            return 0.0
        arr = list(self._windows[patient_id][channel])[-window_steps:]
        valid = [v for v in arr if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    def get_trend(self, patient_id: int, channel: str,
                   window_steps: int = 6) -> float:
        """Return linear slope over the last window_steps readings."""
        if patient_id not in self._windows:
            return 0.0
        arr = np.array(list(self._windows[patient_id][channel])[-window_steps:])
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
