"""Denoising filters for physiological signals."""
import numpy as np


def median_filter(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply median filter to remove spike artefacts."""
    from scipy.signal import medfilt
    return medfilt(signal, kernel_size=window).astype(np.float32)


def bridge_dropouts(signal: np.ndarray, max_gap_samples: int = 2) -> np.ndarray:
    """
    Bridge NaN dropouts up to max_gap_samples using linear interpolation.
    Leaves longer gaps as NaN for downstream handling.
    """
    s = signal.copy().astype(np.float64)
    nans = np.isnan(s)
    if not nans.any():
        return s.astype(np.float32)
    idx = np.arange(len(s))
    # Only interpolate short runs
    x = idx[~nans]
    y = s[~nans]
    if len(x) < 2:
        return s.astype(np.float32)
    # Mark runs longer than max_gap
    nan_runs = []
    in_run = False
    run_start = 0
    for i in range(len(s)):
        if nans[i] and not in_run:
            in_run = True
            run_start = i
        elif not nans[i] and in_run:
            if i - run_start <= max_gap_samples:
                nan_runs.append((run_start, i))
            in_run = False
    interpolated = np.interp(idx, x, y)
    for start, end in nan_runs:
        s[start:end] = interpolated[start:end]
    return s.astype(np.float32)
