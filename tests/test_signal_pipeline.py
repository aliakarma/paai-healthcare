"""Unit tests for Algorithm 1 — Signal Preprocessing."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_denoise_removes_spikes():
    from preprocessing.denoise import median_filter

    s = np.array([120.0, 119.0, 250.0, 121.0, 120.0])  # 250 is spike
    result = median_filter(s, window=3)
    assert result[2] < 200, f"Spike not removed: {result[2]}"


def test_bridge_dropouts():
    from preprocessing.denoise import bridge_dropouts

    s = np.array([120.0, np.nan, np.nan, 121.0, 120.0])
    result = bridge_dropouts(s, max_gap_samples=2)
    assert not np.isnan(result[1])
    assert not np.isnan(result[2])


def test_normalise_channels():
    from preprocessing.normalise import ChannelNormaliser

    norm = ChannelNormaliser()
    z = norm.normalise("sbp", np.array([130.0]))
    assert abs(float(z[0])) < 0.5, "SBP at mean should z-score near 0"


def test_feature_extraction():
    from preprocessing.feature_extraction import rolling_mean, rolling_slope

    s = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    m = rolling_mean(s, 3)
    assert len(m) == 5
    slope = rolling_slope(s, 3)
    assert slope[-1] > 0, "Upward trend should yield positive slope"


def test_signal_pipeline_run():
    from preprocessing.signal_pipeline import SignalPipeline

    pipeline = SignalPipeline()
    vitals = {"sbp": 130, "dbp": 82, "glucose_mgdl": 100, "heart_rate": 72, "spo2": 97}
    x, E = pipeline.run(vitals)
    assert x.shape == (5,)
    assert isinstance(E, list)


def test_anomaly_detection_escalation():
    from preprocessing.signal_pipeline import SignalPipeline

    pipeline = SignalPipeline()
    vitals = {"sbp": 195, "dbp": 115, "glucose_mgdl": 100, "heart_rate": 72, "spo2": 97}
    x, E = pipeline.run(vitals)
    channels = [ev["channel"] for ev in E]
    assert "sbp" in channels, "Hypertensive urgency not flagged"


def test_hypoglycemia_detection():
    from preprocessing.signal_pipeline import SignalPipeline

    pipeline = SignalPipeline()
    vitals = {"sbp": 120, "dbp": 78, "glucose_mgdl": 48, "heart_rate": 90, "spo2": 97}
    _, E = pipeline.run(vitals)
    levels = {ev["channel"]: ev["level"] for ev in E}
    assert "glucose" in levels
    assert levels["glucose"] == "emergency"
