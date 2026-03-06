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
        """Normalize channel values to zero mean and unit variance.
        
        Parameters
        ----------
        channel : str
            Name of the physiological channel.
        values : np.ndarray
            Input values to normalize.
        
        Returns
        -------
        np.ndarray
            Normalized values as float32 array.
        """
        mu = self.means.get(channel, 0.0)
        sigma = self.stds.get(channel, 1.0)
        return ((values - mu) / sigma).astype(np.float32)

    def denormalise(self, channel: str, values: np.ndarray) -> np.ndarray:
        """Reverse normalization to original scale.
        
        Parameters
        ----------
        channel : str
            Name of the physiological channel.
        values : np.ndarray
            Normalized values.
        
        Returns
        -------
        np.ndarray
            Denormalized values as float32 array.
        """
        mu = self.means.get(channel, 0.0)
        sigma = self.stds.get(channel, 1.0)
        return (values * sigma + mu).astype(np.float32)
