"""Unit normalisation and z-score scaling for physiological channels."""

from pathlib import Path

import numpy as np
import yaml


class ChannelNormaliser:
    """Z-score normaliser with per-channel statistics from config."""

    def __init__(self, config_path: str = "configs/preprocessing.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        norm = cfg["normalisation"]
        self.means = norm["channel_means"]
        self.stds = norm["channel_stds"]

    def normalise(self, channel: str, values: np.ndarray) -> np.ndarray:
        mu = self.means.get(channel, 0.0)
        sigma = self.stds.get(channel, 1.0)
        return ((values - mu) / sigma).astype(np.float32)

    def denormalise(self, channel: str, values: np.ndarray) -> np.ndarray:
        mu = self.means.get(channel, 0.0)
        sigma = self.stds.get(channel, 1.0)
        return (values * sigma + mu).astype(np.float32)
