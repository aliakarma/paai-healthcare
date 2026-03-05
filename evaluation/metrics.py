"""metrics.py — All evaluation metrics for Table 2 and supporting figures."""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_precision_recall_f1(y_true: np.ndarray,
                                  y_score: np.ndarray,
                                  threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_latency_cdf(latencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_lat = np.sort(latencies)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    return sorted_lat, cdf


def compute_adherence(adherence_arr: np.ndarray) -> float:
    return float(np.mean(adherence_arr))


def bootstrap_ci(metric_arr: np.ndarray,
                  ci: float = 0.95) -> tuple[float, float]:
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return float(np.percentile(metric_arr, lo)), float(np.percentile(metric_arr, hi))
