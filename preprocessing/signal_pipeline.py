"""
signal_pipeline.py
==================
Algorithm 1 from the paper: Signal Preprocessing and Anomaly Detection.

Require: Raw streams S from IoT, EHR, self-reports
Ensure:  Feature vector x, anomaly set E
"""

from typing import Optional

import numpy as np
import yaml

from preprocessing.denoise import bridge_dropouts, median_filter
from preprocessing.feature_extraction import extract_all_features
from preprocessing.normalise import ChannelNormaliser

CHANNELS = ["sbp", "dbp", "glucose_mgdl", "heart_rate", "spo2"]


class SignalPipeline:
    """
    End-to-end signal preprocessing pipeline.
    Denoises → normalises → featurises → gates anomalies.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "configs/preprocessing.yaml",
    ):
        if config is None:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.normaliser = ChannelNormaliser(config_path)
        with open("configs/escalation_thresholds.yaml") as f:
            self.thresholds = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Algorithm 1 — Step 2: Denoise
    # ------------------------------------------------------------------
    def denoise(self, signal: np.ndarray) -> np.ndarray:
        bridged = bridge_dropouts(
            signal, max_gap_samples=self.cfg["denoise"]["dropout_bridge_seconds"] // 5
        )
        return median_filter(bridged, window=self.cfg["denoise"]["window_points"])

    # ------------------------------------------------------------------
    # Algorithm 1 — Step 3: Normalise units (z-score per channel)
    # ------------------------------------------------------------------
    def normalise(self, channel: str, signal: np.ndarray) -> np.ndarray:
        return self.normaliser.normalise(channel, signal)

    def zscore(self, raw_vector: np.ndarray) -> np.ndarray:
        """Z-score a raw 5-dim vital vector [sbp, dbp, glc, hr, spo2]."""
        out = np.zeros_like(raw_vector, dtype=np.float32)
        for i, ch in enumerate(CHANNELS):
            out[i] = float(self.normaliser.normalise(ch, np.array([raw_vector[i]]))[0])
        return out

    # ------------------------------------------------------------------
    # Algorithm 1 — Step 4: Featurise
    # ------------------------------------------------------------------
    def featurise(self, channel: str, values: np.ndarray) -> dict:
        windows = self.cfg["features"]["rolling_windows_minutes"]
        steps = [max(1, w // 5) for w in windows]  # convert minutes to 5-min steps
        return extract_all_features(channel, values, windows=steps)

    # ------------------------------------------------------------------
    # Algorithm 1 — Steps 6-7: Aggregate + Gate anomalies
    # ------------------------------------------------------------------
    def gate_anomalies(self, vitals: dict) -> list:
        """
        Check current vital readings against clinical thresholds.
        Returns list of anomaly event dicts.
        """
        anomalies = []
        bp_thr = self.thresholds["blood_pressure"]
        glc_thr = self.thresholds["glucose"]
        hr_thr = self.thresholds["heart_rate"]
        spo2_thr = self.thresholds["spo2"]

        sbp = vitals.get("sbp", 0)
        glc = vitals.get("glucose_mgdl", 100)
        hr = vitals.get("heart_rate", 70)
        spo2 = vitals.get("spo2", 98)

        if sbp >= bp_thr["systolic_emergency"]:
            anomalies.append({"channel": "sbp", "level": "emergency", "value": sbp})
        elif sbp >= bp_thr["systolic_urgent"]:
            anomalies.append({"channel": "sbp", "level": "urgent", "value": sbp})

        if glc <= glc_thr["hypoglycemia_severe"]:
            anomalies.append({"channel": "glucose", "level": "emergency", "value": glc})
        elif glc <= glc_thr["hypoglycemia_mild"]:
            anomalies.append({"channel": "glucose", "level": "urgent", "value": glc})
        elif glc >= glc_thr["hyperglycemia_emergency"]:
            anomalies.append({"channel": "glucose", "level": "emergency", "value": glc})
        elif glc >= glc_thr["hyperglycemia_alert"]:
            anomalies.append({"channel": "glucose", "level": "urgent", "value": glc})

        if hr <= hr_thr["bradycardia"] or hr >= hr_thr["tachycardia"]:
            anomalies.append({"channel": "heart_rate", "level": "urgent", "value": hr})

        if spo2 <= spo2_thr["emergency"]:
            anomalies.append({"channel": "spo2", "level": "emergency", "value": spo2})
        elif spo2 <= spo2_thr["alert"]:
            anomalies.append({"channel": "spo2", "level": "urgent", "value": spo2})

        return anomalies

    def run(self, raw_vitals: dict) -> tuple[np.ndarray, list]:
        """
        Full Algorithm 1 pipeline for a single timestep's vital readings.

        Parameters
        ----------
        raw_vitals : dict  e.g. {"sbp": 148.0, "glucose_mgdl": 95.0, ...}

        Returns
        -------
        x : np.ndarray  — 5-dim z-scored feature vector
        E : list        — anomaly set (may be empty)
        """
        x_raw = np.array([raw_vitals.get(ch, 0.0) for ch in CHANNELS], dtype=np.float32)
        x = self.zscore(x_raw)
        E = self.gate_anomalies(raw_vitals)
        return x, E
