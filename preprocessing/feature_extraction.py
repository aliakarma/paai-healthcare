"""
feature_extraction.py
=====================
Extracts rolling statistical features from raw physiological signals.
Implements the featurisation step of Algorithm 1 (paper Section 3.4).
"""
import numpy as np
import pandas as pd


def rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(series)
    return s.rolling(window, min_periods=1).mean().to_numpy(dtype=np.float32)


def rolling_slope(series: np.ndarray, window: int) -> np.ndarray:
    """Estimate local slope via linear regression on rolling window."""
    slopes = np.zeros(len(series), dtype=np.float32)
    for i in range(len(series)):
        start = max(0, i - window + 1)
        segment = series[start:i + 1]
        if len(segment) >= 2:
            x = np.arange(len(segment), dtype=np.float32)
            slopes[i] = float(np.polyfit(x, segment, 1)[0])
    return slopes


def rolling_volatility(series: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(series)
    return s.rolling(window, min_periods=1).std(ddof=0).fillna(0).to_numpy(
        dtype=np.float32)


def rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(series)
    mu  = s.rolling(window, min_periods=1).mean()
    std = s.rolling(window, min_periods=1).std(ddof=0).replace(0, 1)
    return ((s - mu) / std).fillna(0).to_numpy(dtype=np.float32)


def extract_all_features(channel_name: str, values: np.ndarray,
                          windows: list[int] = [1, 3, 12]) -> dict:
    """
    Extract full feature dict for one channel.
    Windows are in units of timesteps (5 min each): [1, 3, 12] = [5, 15, 60 min].
    """
    feats = {}
    for w in windows:
        feats[f"{channel_name}_mean_{w}"] = rolling_mean(values, w)
        feats[f"{channel_name}_slope_{w}"] = rolling_slope(values, w)
    feats[f"{channel_name}_volatility"] = rolling_volatility(values, windows[-1])
    feats[f"{channel_name}_zscore"]     = rolling_zscore(values, windows[-1])
    return feats
