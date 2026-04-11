"""metrics.py — All evaluation metrics for Table 2 and supporting figures."""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute fraction of correctly classified samples."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def compute_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for binary predictions.

    If ``y_pred`` contains continuous scores, they are thresholded at
    ``threshold`` before computing the metrics.
    """
    y_pred_arr = np.asarray(y_pred)
    # Convert continuous scores to binary if needed
    if y_pred_arr.dtype.kind == "f" and not np.all(np.isin(y_pred_arr, [0, 1])):
        y_pred_arr = (y_pred_arr >= threshold).astype(int)
    prec = float(precision_score(y_true, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_arr, zero_division=0))
    return prec, rec, f1


def compute_latency_percentiles(
    latencies: np.ndarray,
) -> tuple[float, float, float]:
    """Return the 50th, 95th, and 99th percentile of a latency array."""
    arr = np.asarray(latencies, dtype=float)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def compute_adherence_correlation(
    adherence: np.ndarray, outcomes: np.ndarray
) -> float:
    """Return Pearson r between adherence scores and patient outcomes."""
    r, _ = pearsonr(np.asarray(adherence), np.asarray(outcomes))
    return float(r)


def compute_latency_cdf(latencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_lat = np.sort(latencies)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    return sorted_lat, cdf


def compute_adherence(adherence_arr: np.ndarray) -> float:
    return float(np.mean(adherence_arr))


def bootstrap_ci(metric_arr: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return float(np.percentile(metric_arr, lo)), float(np.percentile(metric_arr, hi))
