"""
human_schedule.py — Baseline B3: Fixed human-designed schedule.
================================================================
Models a clinician who reviews BP once per day (09:00) and issues an
escalation only if SBP > 160 mmHg at that single daily check.  This
simulates the highest-latency baseline: events that occur outside the
daily review window are completely missed until the next scheduled check.

Key corrections vs. previous version
--------------------------------------
* ``med_precision`` is computed from the schedule's binary escalation
  decisions against event ground-truth using ``sklearn.precision_score``,
  not hardcoded as ``[0.75]*20``.
* Latency is the actual time gap (seconds) from event onset to the next
  scheduled 09:00 review where the clinician would detect the exceedance,
  computed per event from the cohort data.  The previous version sampled
  from ``np.random.exponential(9.8)`` with no link to the data.
* ``accuracy`` is computed and bootstrapped from real per-timestep
  predictions, not estimated from a fixed scalar.
* Bootstrap distributions (1000 resamples, seed=42) are returned for all
  metrics for consistency with all other evaluators.
* A ``y_true`` array is stored in the result dict so ``plot_roc.py`` can
  build accurate ROC curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score
from tqdm import tqdm


# ── Schedule parameters ───────────────────────────────────────────────────────
_REVIEW_HOUR   = 9        # Daily review at 09:00 patient local time
_REVIEW_MINUTE = 0
_SBP_THRESHOLD = 160.     # mmHg — clinician escalates if SBP > this at review
_TIMESTEP_MIN  = 5        # minutes per simulation step


def _is_review_time(t_minutes: int) -> bool:
    """Return True if this timestep falls within the daily 09:00 review window."""
    hour_of_day = (t_minutes // 60) % 24
    return hour_of_day == _REVIEW_HOUR


def _score_row(row: dict, rng: np.random.Generator) -> tuple[int, float]:
    """Produce an action and anomaly score for one timestep.

    The human schedule only escalates during the daily review window and
    only when SBP > threshold.  All other timesteps receive a low score.

    Parameters
    ----------
    row : vital-sign dict for one timestep
    rng : pre-seeded Generator for observation noise

    Returns
    -------
    action : int   — 4 (escalate) or 0 (no action)
    score  : float — anomaly score in [0, 1]
    """
    t_min = int(row.get("t_minutes", 0))
    sbp   = float(row.get("sbp", 120.))

    if _is_review_time(t_min) and sbp > _SBP_THRESHOLD:
        base  = 0.70
        action = 4
    else:
        base  = 0.10
        action = 0

    score = float(np.clip(base + rng.normal(0., 0.05), 0., 1.))
    return action, score


def _compute_latency(
    vitals_df: pd.DataFrame,
    event_set: set,
    action_map: dict[tuple[int, int], int],
) -> list[float]:
    """Measure detection latency (seconds) per true event.

    For the human-schedule baseline, latency is the gap between event onset
    and the next daily review at which an escalation is produced.
    Clipped to 3600 s × 24 = 86400 s (1 day) when no review-time escalation
    ever occurs after an event.

    Parameters
    ----------
    vitals_df  : full longitudinal vitals DataFrame
    event_set  : set of (patient_id, t_minutes) event tuples
    action_map : (patient_id, t_minutes) → predicted action
    """
    latencies: list[float] = []
    MAX_LATENCY_SEC = 24. * 3600.   # 24 hours

    for pid, grp in vitals_df.groupby("patient_id"):
        grp = grp.sort_values("t_minutes").reset_index(drop=True)
        event_times = {t for (p, t) in event_set if p == int(pid)}
        if not event_times:
            continue

        for event_t in event_times:
            # Find the first escalation at or after the event
            after = grp[grp["t_minutes"] >= event_t]
            escalated_t = None
            for _, row in after.iterrows():
                if action_map.get((int(pid), int(row["t_minutes"]))) == 4:
                    escalated_t = int(row["t_minutes"])
                    break

            if escalated_t is not None:
                latency_sec = (escalated_t - event_t) * 60.
                latencies.append(min(float(latency_sec), MAX_LATENCY_SEC))
            else:
                # No escalation ever produced → worst-case latency
                latencies.append(MAX_LATENCY_SEC)

    return latencies


def evaluate(cohort_dir: str) -> dict:
    """Evaluate the human-schedule baseline on the synthetic cohort.

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

    rng_noise  = np.random.default_rng(42)
    y_true:    list[int]   = []
    y_score:   list[float] = []
    actions:   list[int]   = []
    action_map: dict       = {}

    for _, row in tqdm(vitals_df.iterrows(), total=len(vitals_df),
                       desc="Evaluating Human-schedule", leave=False):
        pid   = int(row["patient_id"])
        t     = int(row["t_minutes"])
        label = 1 if (pid, t) in event_set else 0

        act, score = _score_row(row.to_dict(), rng_noise)

        y_true.append(label)
        y_score.append(score)
        actions.append(act)
        action_map[(pid, t)] = act

    y_true_arr  = np.array(y_true,  dtype=int)
    y_score_arr = np.array(y_score, dtype=float)
    actions_arr = np.array(actions, dtype=int)

    # ── Latency from actual review-time escalation timing ─────────────────────
    latencies = _compute_latency(vitals_df, event_set, action_map)
    if not latencies:
        latencies = [24. * 3600.]
    latency_arr = np.array(latencies, dtype=float)

    # ── Bootstrap metrics (1000 resamples, seed=42) ───────────────────────────
    rng = np.random.default_rng(42)
    boot_aucs, boot_accs, boot_precs = [], [], []
    n = len(y_true_arr)

    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        yt, ys, pa = y_true_arr[idx], y_score_arr[idx], actions_arr[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, ys))
        boot_accs.append(float(np.mean((ys > 0.5) == yt)))
        esc_pred = (pa == 4).astype(int)
        boot_precs.append(float(precision_score(yt, esc_pred, zero_division=0)))

    return {
        "roc_auc":       np.array(boot_aucs,  dtype=float),
        "roc_scores":    y_score_arr,
        "y_true":        y_true_arr,
        "accuracy":      np.array(boot_accs,  dtype=float),
        "latency":       latency_arr,
        "med_precision": np.array(boot_precs, dtype=float),
    }
