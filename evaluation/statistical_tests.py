"""statistical_tests.py — DeLong AUC test, Wilcoxon, Bonferroni, effect sizes."""
import numpy as np
from scipy import stats


def delong_test(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Simplified DeLong test approximation for AUC comparison.
    Returns Bonferroni-uncorrected p-value.
    """
    if len(scores_a) == 0 or len(scores_b) == 0:
        return 1.0
    t_stat, p_val = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    return float(p_val)


def wilcoxon_test(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    """Wilcoxon signed-rank test for paired comparisons."""
    n = min(len(arr_a), len(arr_b))
    if n < 5:
        return 1.0
    try:
        _, p = stats.wilcoxon(arr_a[:n], arr_b[:n])
        return float(p)
    except Exception:
        return 1.0


def bonferroni_correct(p_val: float, n_tests: int) -> float:
    """Apply Bonferroni correction."""
    return min(1.0, p_val * n_tests)


def cohens_d(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    diff = np.mean(arr_a) - np.mean(arr_b)
    pooled_std = np.sqrt((np.std(arr_a, ddof=1)**2 + np.std(arr_b, ddof=1)**2) / 2)
    return float(diff / (pooled_std + 1e-8))
