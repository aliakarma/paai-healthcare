"""
predictive_only.py — Baseline B2: IsolationForest anomaly detector.
=====================================================================
No RL, no BDI agents. Unsupervised anomaly detection trained on vital-sign
features, evaluated against escalation-event ground truth.

Key corrections vs. previous version
--------------------------------------
* ``med_precision`` is computed from the model's binary escalation decisions
  (``score > 0.5``) against the event ground-truth using
  ``sklearn.precision_score``, not hardcoded as ``[0.79]*20``.
* Latency is measured as the real time delta (seconds) from the first
  high-anomaly-score reading (``score > 0.5``) to the event timestep,
  per event, rather than being sampled from ``np.random.exponential(3.1)``.
* Bootstrap distributions (1000 resamples, seed=42) are returned for all
  metrics, consistent with the other evaluators.
* A shared ``y_true`` array is stored in the result dict (key ``"y_true"``)
  so that ``plot_roc.py`` can build accurate ROC curves without
  reconstructing ground truth independently.
* Training / inference use the same deterministic split (20 % of data,
  ``random_state=42``) as documented in the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score


FEATURE_COLS = [
    "sbp", "dbp", "glucose_mgdl", "heart_rate", "spo2",
    "adherence_med", "adherence_diet",
]

_ESCALATION_SCORE_THRESHOLD = 0.50   # score > this → predicted escalation


def train_model(vitals_df: pd.DataFrame) -> tuple:
    """Fit IsolationForest on a feature-imputed subset.

    Parameters
    ----------
    vitals_df : full (or subset) longitudinal vitals DataFrame.

    Returns
    -------
    model  : fitted IsolationForest
    scaler : fitted StandardScaler
    """
    X = vitals_df[FEATURE_COLS].copy()
    # Impute with column means (consistent with evaluation imputation below)
    X = X.fillna(X.mean())
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators  = 200,
        contamination = 0.05,
        random_state  = 42,
        n_jobs        = -1,
    )
    model.fit(X_scaled)
    return model, scaler


def _anomaly_scores(
    vitals_df: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
) -> np.ndarray:
    """Convert IsolationForest decision_function output to [0, 1] anomaly scores.

    Higher scores → more anomalous.  The raw decision_function output is
    negated and min-max normalised so that the resulting score is
    monotonically increasing with anomaly degree.

    Parameters
    ----------
    vitals_df : full longitudinal DataFrame (all patients)
    model     : fitted IsolationForest
    scaler    : fitted StandardScaler

    Returns
    -------
    scores : 1-D float array, same length as vitals_df
    """
    X = vitals_df[FEATURE_COLS].copy().fillna(
        vitals_df[FEATURE_COLS].mean())
    X_scaled   = scaler.transform(X)
    raw_scores = model.decision_function(X_scaled)  # higher = more normal
    # Invert so that higher value = more anomalous
    inverted   = -raw_scores
    score_min  = inverted.min()
    score_range = inverted.max() - score_min + 1e-8
    return ((inverted - score_min) / score_range).astype(float)


def _compute_latency(
    vitals_df: pd.DataFrame,
    event_set: set,
    scores: np.ndarray,
    threshold: float = _ESCALATION_SCORE_THRESHOLD,
    timestep_minutes: int = 5,
) -> list[float]:
    """Measure detection latency (seconds) per true event.

    Latency = time from first high-score reading (score > threshold) to
    the event timestep.  Clipped at 3600 s when the model never raises
    an alert within the episode.

    Parameters
    ----------
    vitals_df        : longitudinal vitals DataFrame
    event_set        : set of (patient_id, t_minutes) event tuples
    scores           : anomaly score array aligned with vitals_df rows
    threshold        : score above which we consider it a predicted alert
    timestep_minutes : simulation resolution in minutes
    """
    score_series = pd.Series(scores, index=vitals_df.index)
    latencies: list[float] = []

    for pid, grp in vitals_df.groupby("patient_id"):
        grp_scores = score_series.loc[grp.index]
        event_times = {t for (p, t) in event_set if p == int(pid)}
        if not event_times:
            continue

        for event_t in event_times:
            before_mask  = grp["t_minutes"] <= event_t
            grp_before   = grp[before_mask]
            scores_before = grp_scores.loc[grp_before.index]

            alert_mask = scores_before > threshold
            if not alert_mask.any():
                latencies.append(3600.)
                continue

            first_alert_t = int(grp_before.loc[alert_mask.index[alert_mask]].iloc[0]["t_minutes"])
            latency_sec   = (event_t - first_alert_t) * 60.
            latencies.append(min(max(float(latency_sec), 0.), 3600.))

    return latencies


def evaluate(cohort_dir: str) -> dict:
    """Evaluate the IsolationForest baseline on the synthetic cohort.

    Training uses a random 20 % sample (``random_state=42``) to avoid
    look-ahead bias; inference runs on the full dataset.

    Parameters
    ----------
    cohort_dir : directory containing ``vitals_longitudinal.csv``
                 and ``events.csv``.

    Returns
    -------
    dict with keys:
        roc_auc       — np.ndarray, 1000 bootstrap AUC values
        roc_scores    — np.ndarray, per-timestep anomaly scores
        y_true        — np.ndarray, per-timestep binary ground-truth
        accuracy      — np.ndarray, 1000 bootstrap accuracy values
        latency       — np.ndarray, per-event latency in seconds
        med_precision — np.ndarray, 1000 bootstrap precision values
    """
    vitals_df = pd.read_csv(f"{cohort_dir}/vitals_longitudinal.csv")
    events_df  = pd.read_csv(f"{cohort_dir}/events.csv")

    event_set = set(zip(
        events_df["patient_id"].astype(int),
        events_df["t_minutes"].astype(int),
    ))

    # ── Train on 20 % random sample ───────────────────────────────────────────
    print("  Training IsolationForest on 20 % of data …")
    train_sample = vitals_df.sample(frac=0.20, random_state=42)
    model, scaler = train_model(train_sample)

    # ── Score full dataset ────────────────────────────────────────────────────
    scores = _anomaly_scores(vitals_df, model, scaler)

    # ── Ground-truth labels (same row order as vitals_df) ─────────────────────
    y_true = np.array([
        1 if (int(r["patient_id"]), int(r["t_minutes"])) in event_set else 0
        for _, r in vitals_df.iterrows()
    ], dtype=int)

    # ── Latency from actual alert timing ─────────────────────────────────────
    latencies = _compute_latency(vitals_df, event_set, scores)
    if not latencies:
        latencies = [3600.]
    latency_arr = np.array(latencies, dtype=float)

    # ── Bootstrap metrics (1000 resamples, seed=42) ───────────────────────────
    rng = np.random.default_rng(42)
    boot_aucs, boot_accs, boot_precs = [], [], []
    n = len(y_true)

    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, ys))
        boot_accs.append(float(np.mean((ys > _ESCALATION_SCORE_THRESHOLD) == yt)))
        esc_pred = (ys > _ESCALATION_SCORE_THRESHOLD).astype(int)
        boot_precs.append(float(precision_score(yt, esc_pred, zero_division=0)))

    return {
        "roc_auc":       np.array(boot_aucs,  dtype=float),
        "roc_scores":    scores,
        "y_true":        y_true,
        "accuracy":      np.array(boot_accs,  dtype=float),
        "latency":       latency_arr,
        "med_precision": np.array(boot_precs, dtype=float),
    }
