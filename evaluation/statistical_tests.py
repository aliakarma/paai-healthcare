"""
statistical_tests.py
====================
Statistical tests for comparing model AUC scores and effect sizes.

Key corrections vs. previous version
--------------------------------------
* ``delong_test()`` now implements the actual DeLong et al. (1988) method
  for comparing two correlated or independent ROC AUC scores.  The previous
  version ran a Welch two-sample t-test on bootstrap AUC arrays, which is
  statistically incorrect for AUC comparison (different null hypothesis,
  assumes normality of bootstrap samples, ignores correlation structure).
* ``delong_test()`` accepts the raw score arrays and binary ground-truth
  labels directly, mirroring the paper's evaluation protocol.
* A convenience ``delong_test_from_bootstrap()`` is provided for callers
  that only have bootstrap arrays (e.g. legacy code), with a clear docstring
  warning about the approximation.
* ``wilcoxon_test()``, ``bonferroni_correct()``, and ``cohens_d()`` are
  unchanged in semantics but now validated for edge cases.

References
----------
DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988).
    Comparing the areas under two or more correlated receiver operating
    characteristic curves: a nonparametric approach.
    Biometrics, 44(3), 837–845.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

# ── DeLong implementation ─────────────────────────────────────────────────────


def _structural_components(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the AUC and its structural components (V10, V01) for one classifier.

    These components are the building blocks of the DeLong variance estimator.

    Parameters
    ----------
    y_true  : 1-D integer array of binary labels (0 / 1).
    y_score : 1-D float array of predicted scores.

    Returns
    -------
    auc  : float  — Mann-Whitney U-based AUC estimate
    V10  : array  — per-positive placement values (length = n_pos)
    V01  : array  — per-negative placement values (length = n_neg)
    """
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    pos_scores = y_score[pos_mask]  # shape (n_pos,)
    neg_scores = y_score[neg_mask]  # shape (n_neg,)

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive and negative labels.")

    # V10[i] = fraction of negatives that score strictly below positive[i]
    #          + 0.5 * fraction tied  (Wilcoxon kernel)
    V10 = np.array(
        [np.mean(neg_scores < p) + 0.5 * np.mean(neg_scores == p) for p in pos_scores],
        dtype=float,
    )

    # V01[j] = fraction of positives that score strictly above negative[j]
    #          + 0.5 * fraction tied
    V01 = np.array(
        [np.mean(pos_scores > n) + 0.5 * np.mean(pos_scores == n) for n in neg_scores],
        dtype=float,
    )

    auc = float(V10.mean())
    return auc, V10, V01


def delong_test(
    y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray
) -> float:
    """DeLong et al. (1988) test for the equality of two AUC values.

    Tests H₀: AUC_A = AUC_B using the non-parametric variance estimator
    described in DeLong et al. (1988).  Suitable for both correlated
    (same patients) and independent classifiers.

    Parameters
    ----------
    y_true    : 1-D binary label array shared by both classifiers.
    y_score_a : 1-D score array for classifier A (e.g. AgHealth+).
    y_score_b : 1-D score array for classifier B (e.g. a baseline).

    Returns
    -------
    p_value : float — two-tailed p-value for H₀: AUC_A = AUC_B.

    Raises
    ------
    ValueError
        If y_true contains only one class, or if array lengths do not match.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score_a = np.asarray(y_score_a, dtype=float)
    y_score_b = np.asarray(y_score_b, dtype=float)

    if len(y_true) != len(y_score_a) or len(y_true) != len(y_score_b):
        raise ValueError("y_true, y_score_a, and y_score_b must have the same length.")

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "y_true must contain at least one positive and one negative label."
        )

    auc_a, V10_a, V01_a = _structural_components(y_true, y_score_a)
    auc_b, V10_b, V01_b = _structural_components(y_true, y_score_b)

    # ── DeLong covariance matrix (S) ─────────────────────────────────────────
    # S = (1/n_pos) * S10  +  (1/n_neg) * S01
    # where S10 is the 2×2 covariance of (V10_a, V10_b) and similarly for S01.

    def _cov2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """2×2 sample covariance matrix of column vectors u and v."""
        m = np.column_stack([u, v])  # (n, 2)
        return np.cov(m, rowvar=False, ddof=1)  # (2, 2)

    S10 = _cov2(V10_a, V10_b)  # (2, 2)
    S01 = _cov2(V01_a, V01_b)  # (2, 2)

    S = S10 / n_pos + S01 / n_neg  # (2, 2) — DeLong variance matrix

    # ── Test statistic ────────────────────────────────────────────────────────
    # L = [1, -1]  (we test the difference AUC_A - AUC_B)
    L = np.array([1.0, -1.0])
    auc_diff = auc_a - auc_b

    var_diff = float(L @ S @ L)  # scalar variance of (AUC_A - AUC_B)

    if var_diff <= 0.0:
        warnings.warn(
            "DeLong variance estimate is non-positive "
            f"(var={var_diff:.6e}).  Returning p=1.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0

    z_stat = auc_diff / np.sqrt(var_diff)

    # Two-tailed p-value from standard normal (DeLong et al. use N(0,1))
    p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
    return p_value


def delong_test_from_bootstrap(
    boot_aucs_a: np.ndarray,
    boot_aucs_b: np.ndarray,
) -> float:
    """Approximate comparison of two AUC distributions using a paired t-test.

    .. warning::
        This is an **approximation** for use when only bootstrap AUC arrays
        are available (e.g. from pre-computed results).  It is **not** the
        true DeLong test and should not be described as such in a manuscript.
        Prefer :func:`delong_test` whenever the raw score arrays are available.

    Parameters
    ----------
    boot_aucs_a : bootstrap AUC samples for model A  (n=1000 recommended)
    boot_aucs_b : bootstrap AUC samples for model B

    Returns
    -------
    p_value : float — two-tailed p-value
    """
    n = min(len(boot_aucs_a), len(boot_aucs_b))
    if n < 5:
        return 1.0
    _, p_value = stats.ttest_rel(boot_aucs_a[:n], boot_aucs_b[:n])
    return float(p_value)


# ── Supporting tests ──────────────────────────────────────────────────────────


def wilcoxon_test(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    """Wilcoxon signed-rank test for paired comparisons of bootstrap samples.

    Parameters
    ----------
    arr_a, arr_b : paired bootstrap metric arrays (equal or unequal length;
                   the shorter length is used for pairing)

    Returns
    -------
    p_value : float — two-tailed p-value.  Returns 1.0 if sample is too small.
    """
    n = min(len(arr_a), len(arr_b))
    if n < 5:
        return 1.0
    a, b = np.asarray(arr_a[:n], dtype=float), np.asarray(arr_b[:n], dtype=float)
    diffs = a - b
    if np.all(diffs == 0.0):
        return 1.0  # identical — trivially equal
    try:
        _, p = stats.wilcoxon(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return 1.0


def bonferroni_correct(p_val: float, n_tests: int) -> float:
    """Apply Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_val   : uncorrected p-value
    n_tests : number of simultaneous comparisons

    Returns
    -------
    corrected p-value, capped at 1.0
    """
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    return float(min(1.0, p_val * n_tests))


def cohens_d(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two independent samples.

    Parameters
    ----------
    arr_a, arr_b : 1-D float arrays

    Returns
    -------
    Cohen's d (positive = arr_a > arr_b)
    """
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)
