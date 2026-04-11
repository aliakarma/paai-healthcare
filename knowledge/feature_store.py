"""
feature_store.py
================
Rolling feature cache for the RL policy state construction.
Stores recent vital sign windows and pre-computes trend features.
"""

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


class FeatureStore:
    """Maintains a rolling window of recent vital readings per patient."""

    CHANNELS = ["sbp", "dbp", "glucose_mgdl", "heart_rate", "spo2"]

    def __init__(self, config_path: str = "configs/preprocessing.yaml"):
        """Initialize feature store with max window size from config.
        
        Parameters
        ----------
        config_path : str
            Path to preprocessing configuration file.
        """
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.max_window = cfg.get("feature_store", {}).get("max_window_samples", 720)
        self._windows: dict[int, dict[str, deque]] = {}

    def push(self, patient_id: int, vital: dict) -> None:
        """Add a new vital reading for a patient.
        
        Parameters
        ----------
        patient_id : int
            Patient identifier.
        vital : dict
            Dictionary of vital signs for current timestep.
        """
        if patient_id not in self._windows:
            self._windows[patient_id] = {
                ch: deque(maxlen=self.max_window) for ch in self.CHANNELS
            }
        for ch in self.CHANNELS:
            self._windows[patient_id][ch].append(vital.get(ch, np.nan))

    def get_rolling_mean(
        self, patient_id: int, channel: str, window_steps: int = 12
    ) -> float:
        if patient_id not in self._windows:
            return 0.0
        arr = list(self._windows[patient_id][channel])[-window_steps:]
        valid = [v for v in arr if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    def get_trend(self, patient_id: int, channel: str, window_steps: int = 6) -> float:
        """Return linear slope over the last window_steps readings."""
        if patient_id not in self._windows:
            return 0.0
        arr = np.array(list(self._windows[patient_id][channel])[-window_steps:])
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
