"""
rules_only.py — Baseline B1: Threshold-only rule engine.
=========================================================
No learning, no knowledge graph, no RL. Pure if-then clinical rules based
on published AHA/ADA 2023 escalation thresholds.

Key corrections vs. previous version
--------------------------------------
* ``med_precision`` is now computed from the rule engine's predicted escalation
  decisions against the event ground-truth using ``sklearn.precision_score``,
  not returned as a hardcoded ``[0.71]*20`` array.
* Latency is measured as the real time delta (seconds) between the first
  threshold-crossing reading and the timestep at which the rule engine
  first outputs ``action==4`` (escalate), per event.  It was previously
  sampled from ``np.random.exponential(4.9)`` with no connection to the data.
* ``accuracy`` is computed as ``np.mean((score > 0.5) == y_true)`` from the
  actual per-timestep scores, then bootstrapped over 1000 resamples.
* Bootstrap distributions (1000 resamples, seed=42) are returned for all
  metrics so that ``run_evaluation.py`` can compute 95% CIs consistently
  with the other methods.
* The ``predict()`` function is unchanged — it remains the canonical
  threshold-rule implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score
from tqdm import tqdm

# ── Thresholds (AHA/ADA 2023 — mirrors escalation_criteria.json) ─────────────
_AUTO_SBP = 180  # mmHg — automatic escalation
_WATCH_SBP = 160  # mmHg — watch zone
_AUTO_GLU_LO = 54  # mg/dL — severe hypoglycaemia
_WATCH_GLU_LO = 70  # mg/dL — mild hypoglycaemia
_AUTO_GLU_HI = 400  # mg/dL — hyperglycaemic emergency
_WATCH_GLU_HI = 300  # mg/dL — hyperglycaemic alert
_AUTO_SPO2 = 90  # % — emergency
_WATCH_SPO2 = 93  # % — alert
_LOW_ADHERENCE_THRESHOLD = 0.50


def predict(vitals: dict) -> tuple[int, float]:
    """Apply threshold rules to one vital-sign snapshot.

    Parameters
    ----------
    vitals : dict with keys ``sbp``, ``glucose_mgdl``, ``spo2``,
             ``heart_rate``, ``adherence_med``.

    Returns
    -------
    action : int  — 0 (no action), 1 (med reminder), 4 (escalate)
    score  : float — continuous anomaly score in [0, 1]
    """
    sbp = float(vitals.get("sbp", 120.0))
    glc = float(vitals.get("glucose_mgdl", 100.0))
    spo2 = float(vitals.get("spo2", 98.0))
    adh = float(vitals.get("adherence_med", 0.7))

    # Automatic escalation zone
    if (
        sbp >= _AUTO_SBP
        or glc <= _AUTO_GLU_LO
        or glc >= _AUTO_GLU_HI
        or spo2 <= _AUTO_SPO2
    ):
        return 4, 0.92

    # Watch zone (elevated but not yet automatic)
    if (
        sbp >= _WATCH_SBP
        or glc <= _WATCH_GLU_LO
        or glc >= _WATCH_GLU_HI
        or spo2 <= _WATCH_SPO2
    ):
        return 4, 0.65

    # Low adherence → medication reminder
    if adh < _LOW_ADHERENCE_THRESHOLD:
        return 1, 0.30

    return 0, 0.05


def _compute_latency(
    vitals_df: pd.DataFrame,
    event_set: set,
    action_map: dict[tuple[int, int], int],
    timestep_minutes: int = 5,
) -> list[float]:
    """Measure detection latency (seconds) per true event.

    For each event, latency = time from first threshold-crossing reading
    to the first timestep at which the rule engine outputs action 4.
    Clipped to 3600 s (1 hour) when no escalation is ever produced.

    Parameters
    ----------
    vitals_df       : full longitudinal vitals DataFrame
    event_set       : set of (patient_id, t_minutes) tuples marking events
    action_map      : (patient_id, t_minutes) → predicted action
    timestep_minutes: resolution of the simulation (default 5 min)
    """
    latencies: list[float] = []

    for pid, grp in vitals_df.groupby("patient_id"):
        grp = grp.sort_values("t_minutes").reset_index(drop=True)
        event_times = {t for (p, t) in event_set if p == int(pid)}
        if not event_times:
            continue

        for event_t in event_times:
            # First crossing at or before the event
            before = grp[grp["t_minutes"] <= event_t]
            if before.empty:
                continue

            crossings = before[
                before.apply(lambda r: predict(r.to_dict())[0] == 4, axis=1)
            ]
            if crossings.empty:
                latencies.append(3600.0)
                continue

            first_cross_t = int(crossings["t_minutes"].iloc[0])

            # First escalation action at or after first crossing
            after = grp[grp["t_minutes"] >= first_cross_t]
            escalated_t = None
            for _, row in after.iterrows():
                if action_map.get((int(pid), int(row["t_minutes"]))) == 4:
                    escalated_t = int(row["t_minutes"])
                    break

            if escalated_t is not None:
                latency_sec = (escalated_t - first_cross_t) * 60.0
            else:
                latency_sec = 3600.0

            latencies.append(min(float(latency_sec), 3600.0))

    return latencies


def evaluate(cohort_dir: str) -> dict:
    """Evaluate the rules-only baseline on the synthetic cohort.

    Parameters
    ----------
    cohort_dir : directory containing ``vitals_longitudinal.csv``
                 and ``events.csv``.

    Returns
    -------
    dict with keys:
        roc_auc       — np.ndarray, 1000 bootstrap AUC values
        roc_scores    — np.ndarray, per-timestep anomaly scores
        accuracy      — np.ndarray, 1000 bootstrap accuracy values
        latency       — np.ndarray, per-event latency in seconds
        med_precision — np.ndarray, 1000 bootstrap precision values
    """
    vitals_df = pd.read_csv(f"{cohort_dir}/vitals_longitudinal.csv")
    events_df = pd.read_csv(f"{cohort_dir}/events.csv")

    event_set = set(
        zip(
            events_df["patient_id"].astype(int),
            events_df["t_minutes"].astype(int),
        )
    )

    y_true: list[int] = []
    y_score: list[float] = []
    actions: list[int] = []
    action_map: dict = {}

    rng_noise = np.random.default_rng(42)

    for _, row in tqdm(
        vitals_df.iterrows(),
        total=len(vitals_df),
        desc="Evaluating Rules-only",
        leave=False,
    ):
        pid = int(row["patient_id"])
        t = int(row["t_minutes"])
        label = 1 if (pid, t) in event_set else 0

        act, base_score = predict(row.to_dict())
        # Small observation noise so AUC is not degenerate
        score = float(np.clip(base_score + rng_noise.normal(0.0, 0.04), 0.0, 1.0))

        y_true.append(label)
        y_score.append(score)
        actions.append(act)
        action_map[(pid, t)] = act

    y_true_arr = np.array(y_true, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)
    actions_arr = np.array(actions, dtype=int)

    # ── Latency from actual threshold crossings ───────────────────────────────
    latencies = _compute_latency(vitals_df, event_set, action_map)
    if not latencies:
        latencies = [3600.0]
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
        "roc_auc": np.array(boot_aucs, dtype=float),
        "roc_scores": y_score_arr,
        "y_true": y_true_arr,
        "accuracy": np.array(boot_accs, dtype=float),
        "latency": latency_arr,
        "med_precision": np.array(boot_precs, dtype=float),
    }
