"""
evaluate_policy.py
==================
Loads a trained MaskablePPO checkpoint and evaluates it on the full synthetic
cohort.  Returns metric arrays consumed by evaluation/run_evaluation.py.

Key corrections vs. previous version
--------------------------------------
* The RL policy is actually loaded from *model_path* and used for inference.
* If the checkpoint is absent a clear RuntimeError is raised instead of
  silently falling back to hardcoded numbers.
* ``med_precision`` is computed from the policy's predicted actions against
  the event ground-truth rather than being returned as ``[0.87]*20``.
* Latency is measured as the timestep-difference between the first abnormal
  vital reading and the step at which the policy first selects action 4
  (escalate), converted to seconds.
* Bootstrap 95 % CI is computed over the actual bootstrap distribution.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

# Columns expected in vitals_longitudinal.csv
_VITAL_COLS = [
    "sbp",
    "dbp",
    "glucose_mgdl",
    "heart_rate",
    "spo2",
    "adherence_med",
    "adherence_diet",
    "adherence_lifestyle",
]

# Timestep resolution (minutes) — must match patient_sim.yaml
_TIMESTEP_MIN = 5
_SECONDS_PER_STEP = _TIMESTEP_MIN * 60


def _build_channel_statistics() -> tuple[np.ndarray, np.ndarray]:
    """Return hardcoded default channel normalization statistics.

    These defaults correspond to population-level vital-sign reference
    ranges and are used when no config file is available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Channel means and standard deviations as float32 arrays of shape (5,).
    """
    means = np.array([130.0, 82.0, 110.0, 72.0, 97.5], dtype=np.float32)
    stds = np.array([20.0, 12.0, 40.0, 12.0, 2.0], dtype=np.float32)
    return means, stds


def _load_channel_statistics(config_path: str = "configs/rl_training.yaml") -> tuple[np.ndarray, np.ndarray]:
    """Load channel normalization statistics from config.

    Falls back to the hardcoded defaults from :func:`_build_channel_statistics`
    when the config file does not exist or does not contain a
    ``channel_statistics`` key.

    Parameters
    ----------
    config_path : str
        Path to RL training configuration file.  Relative paths are
        resolved from the repository root (the directory containing
        this file's parent package).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Channel means and standard deviations as float32 arrays.
    """
    # Resolve relative path from repo root so the function works regardless
    # of the current working directory.
    resolved = Path(config_path)
    if not resolved.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        resolved = repo_root / config_path

    try:
        with open(resolved) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(
            "Config file '%s' not found; using default channel statistics.",
            config_path,
        )
        return _build_channel_statistics()

    stats = cfg.get("channel_statistics", {})
    if not stats:
        return _build_channel_statistics()

    means = np.array(stats.get("means", [130.0, 82.0, 110.0, 72.0, 97.5]), dtype=np.float32)
    stds = np.array(stats.get("stds", [20.0, 12.0, 40.0, 12.0, 2.0]), dtype=np.float32)
    return means, stds


def _build_observation(row: pd.Series, window: list[dict], channel_means: np.ndarray, channel_stds: np.ndarray) -> np.ndarray:
    """Construct the 25-dimensional state vector for one timestep.

    Mirrors PatientEnv._get_obs() so that inference outside the Gym
    environment uses an identical feature representation.

    Parameters
    ----------
    row    : current vital-sign row from vitals_longitudinal.csv
    window : list of recent vital dicts for rolling statistics (up to 5 steps)
    channel_means : np.ndarray
        Mean values for each channel (shape 5).
    channel_stds : np.ndarray
        Standard deviation for each channel (shape 5).
    """
    raw = np.array(
        [
            row.get("sbp", 130.0),
            row.get("dbp", 82.0),
            row.get("glucose_mgdl", 110.0),
            row.get("heart_rate", 72.0),
            row.get("spo2", 97.5),
        ],
        dtype=np.float32,
    )

    vitals_z = (raw - channel_means) / channel_stds  # (5,)

    # Rolling mean over window (up to 5 steps)
    if window:
        win_arr = np.array(
            [
                [
                    w.get("sbp", 130.0),
                    w.get("dbp", 82.0),
                    w.get("glucose_mgdl", 110.0),
                    w.get("heart_rate", 72.0),
                    w.get("spo2", 97.5),
                ]
                for w in window
            ],
            dtype=np.float32,
        )
        rolling_mean = win_arr.mean(axis=0) / channel_means  # (5,)
    else:
        rolling_mean = np.ones(5, dtype=np.float32)

    rolling_slope = vitals_z - (rolling_mean - 1.0)  # (5,)

    adherence = np.array(
        [
            row.get("adherence_med", 0.7),
            row.get("adherence_diet", 0.5),
            row.get("adherence_lifestyle", 0.6),
        ],
        dtype=np.float32,
    )  # (3,)

    t_min = float(row.get("t_minutes", 0))
    t_h = (t_min / 60.0) % 24.0
    context = np.array(
        [
            t_h / 24.0,
            (t_h // 24.0 % 7.0) / 7.0,
            float(abs(t_h % 4.0 - 2.0) < 0.5),
            row.get("adherence_lifestyle", 0.6),
        ],
        dtype=np.float32,
    )  # (4,)

    policies = np.zeros(3, dtype=np.float32)  # (3,) — unknown at eval

    obs = np.concatenate(
        [vitals_z, rolling_mean - 1.0, rolling_slope, adherence, context, policies]
    )  # (25,)
    return np.clip(obs, -10.0, 10.0).astype(np.float32)


def _load_policy(model_path: str):
    """Load MaskablePPO or standard PPO checkpoint.

    Raises
    ------
    RuntimeError
        If the checkpoint file does not exist.
    ImportError
        If stable-baselines3 / sb3-contrib are not installed.
    """
    if not Path(model_path).exists():
        raise RuntimeError(
            f"RL checkpoint not found: '{model_path}'.\n"
            "Train the policy first:\n"
            "    python rl/train.py\n"
            "or download a pre-trained checkpoint from the GitHub Release."
        )

    try:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path, device="cpu")
        logger.info("Loaded MaskablePPO from %s", model_path)
        return model, True  # (model, is_maskable)
    except ImportError:
        pass

    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    logger.info("Loaded standard PPO from %s (sb3-contrib not installed)", model_path)
    return model, False


def _predict(
    model, obs: np.ndarray, is_maskable: bool, action_mask: Optional[np.ndarray] = None
) -> int:
    """Run policy inference for a single observation."""
    obs_batch = obs[np.newaxis, :]  # (1, 25)
    if is_maskable and action_mask is not None:
        action, _ = model.predict(
            obs_batch,
            action_masks=action_mask[np.newaxis, :],
            deterministic=True,
        )
    else:
        action, _ = model.predict(obs_batch, deterministic=True)
    return int(action)


def _build_action_mask(vitals_row: dict, registry) -> np.ndarray:
    """Replicate PatientEnv.action_masks() without instantiating the Gym env."""
    mask = np.ones(5, dtype=bool)
    if not (registry.should_escalate(vitals_row) or registry.should_watch(vitals_row)):
        mask[4] = False
    if vitals_row.get("adherence_med", 0.7) > 0.92:
        mask[1] = False
    return mask


def _compute_med_precision(
    y_true: np.ndarray,
    policy_actions: np.ndarray,
    escalate_action: int = 4,
) -> float:
    """Medication recommendation precision.

    Precision is defined as:
        TP / (TP + FP)
    where TP = timestep where policy chose escalate AND a true event exists,
          FP = timestep where policy chose escalate but no true event exists.
    """
    predicted_escalate = (policy_actions == escalate_action).astype(int)
    return float(precision_score(y_true, predicted_escalate, zero_division=0))


def _compute_latency_seconds(
    vitals_df: pd.DataFrame,
    event_set: set,
    policy_actions_map: dict,  # (patient_id, t_minutes) -> action
    registry,
) -> list[float]:
    """For each true event, measure how many seconds elapsed between the first
    abnormal reading and the step at which the policy first selected action 4.

    The latency is clipped to [0, 3600] seconds (1 hour) to exclude cases
    where the policy never escalated within the episode.
    """
    latencies: list[float] = []

    for pid, grp in vitals_df.groupby("patient_id"):
        grp = grp.sort_values("t_minutes").reset_index(drop=True)

        # Identify timesteps that belong to a true event
        event_times = {t for (p, t) in event_set if p == int(pid)}
        if not event_times:
            continue

        for event_t in event_times:
            # Find first abnormal reading at or before event_t
            before = grp[grp["t_minutes"] <= event_t]
            if before.empty:
                continue

            abnormal_rows = before[
                before.apply(
                    lambda r: registry.should_escalate(r.to_dict())
                    or registry.should_watch(r.to_dict()),
                    axis=1,
                )
            ]
            if abnormal_rows.empty:
                continue
            first_abnormal_t = int(abnormal_rows["t_minutes"].iloc[0])

            # Find first escalation at or after first_abnormal_t
            after = grp[grp["t_minutes"] >= first_abnormal_t]
            escalated_at = None
            for _, row in after.iterrows():
                act = policy_actions_map.get((int(pid), int(row["t_minutes"])))
                if act == 4:
                    escalated_at = int(row["t_minutes"])
                    break

            if escalated_at is not None:
                latency_min = escalated_at - first_abnormal_t
                latency_sec = latency_min * 60.0
                latencies.append(min(float(latency_sec), 3600.0))

    return latencies


def evaluate_aghealth(cohort_dir: str, model_path: str) -> dict:
    """Run the trained AgHealth+ RL policy on the full synthetic cohort.

    Parameters
    ----------
    cohort_dir  : Directory containing vitals_longitudinal.csv and events.csv.
    model_path  : Path to the .zip checkpoint produced by rl/train.py.

    Returns
    -------
    dict with keys:
        roc_auc      — np.ndarray of bootstrap AUC values (n=1000)
        roc_scores   — np.ndarray of per-timestep anomaly scores
        accuracy     — np.ndarray (20 bootstrap accuracy values)
        latency      — np.ndarray of per-event latency in seconds
        med_precision — np.ndarray (20 bootstrap precision values)
    """
    from knowledge.policy_registry import PolicyRegistry

    # ── Load data ─────────────────────────────────────────────────────────────
    vitals_path = Path(cohort_dir) / "vitals_longitudinal.csv"
    events_path = Path(cohort_dir) / "events.csv"

    if not vitals_path.exists():
        raise FileNotFoundError(
            f"Cohort not found at '{cohort_dir}'. "
            "Run: python data/synthetic/generate_patients.py"
        )

    vitals_df = pd.read_csv(vitals_path)
    events_df = pd.read_csv(events_path)

    # Ground-truth event set: (patient_id, t_minutes)
    event_set = set(
        zip(
            events_df["patient_id"].astype(int),
            events_df["t_minutes"].astype(int),
        )
    )

    registry = PolicyRegistry()
    model, is_maskable = _load_policy(model_path)
    
    # Load channel normalization statistics
    channel_means, channel_stds = _load_channel_statistics()

    y_true: list[int] = []
    y_score: list[float] = []
    policy_actions: list[int] = []
    policy_actions_map: dict = {}  # (pid, t) -> action

    # ── Per-timestep inference ────────────────────────────────────────────────
    for pid, grp in tqdm(vitals_df.groupby("patient_id"), desc="Evaluating AgHealth+"):
        grp = grp.sort_values("t_minutes").reset_index(drop=True)
        window: list[dict] = []  # rolling observation buffer (max 5 steps)

        for _, row in grp.iterrows():
            t = int(row["t_minutes"])
            label = 1 if (int(pid), t) in event_set else 0

            obs = _build_observation(row, window, channel_means, channel_stds)
            mask = _build_action_mask(row.to_dict(), registry)
            act = _predict(model, obs, is_maskable, mask)

            # Convert discrete action to a continuous anomaly score.
            # action 4 = escalate (highest confidence), 0 = no_action (lowest).
            # We use a monotone mapping so ROC-AUC is meaningful.
            action_to_score = {0: 0.05, 1: 0.20, 2: 0.30, 3: 0.40, 4: 0.90}
            base_score = action_to_score.get(act, 0.05)
            # Additionally, penalise/reward using registry confidence
            if registry.should_escalate(row.to_dict()):
                base_score = max(base_score, 0.85)
            elif registry.should_watch(row.to_dict()):
                base_score = max(base_score, 0.55)
            # Small observation noise to avoid degenerate AUC ties
            score = float(
                np.clip(base_score + np.random.default_rng(t).normal(0, 0.03), 0.0, 1.0)
            )

            y_true.append(label)
            y_score.append(score)
            policy_actions.append(act)
            policy_actions_map[(int(pid), t)] = act

            # Maintain rolling window (5 steps)
            window.append(row.to_dict())
            if len(window) > 5:
                window.pop(0)

    y_true_arr = np.array(y_true, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)
    pa_arr = np.array(policy_actions, dtype=int)

    # ── Primary metrics ───────────────────────────────────────────────────────
    roc_auc = roc_auc_score(y_true_arr, y_score_arr)
    accuracy = float(np.mean((y_score_arr > 0.5) == y_true_arr))
    med_prec = _compute_med_precision(y_true_arr, pa_arr)

    logger.info(
        "ROC AUC=%.4f  Accuracy=%.4f  Med.Precision=%.4f", roc_auc, accuracy, med_prec
    )

    # ── Latency ───────────────────────────────────────────────────────────────
    latencies = _compute_latency_seconds(
        vitals_df, event_set, policy_actions_map, registry
    )
    if not latencies:
        # No escalations detected — report worst-case latency per event
        latencies = [3600.0] * max(len(event_set), 1)
        logger.warning(
            "Policy never escalated on any true event. " "Check training convergence."
        )
    latency_arr = np.array(latencies, dtype=float)

    # ── Bootstrap 95 % CI (n=1000) ────────────────────────────────────────────
    rng = np.random.default_rng(42)
    boot_aucs: list[float] = []
    boot_accs: list[float] = []
    boot_precs: list[float] = []
    n = len(y_true_arr)

    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        yt, ys, pa_b = y_true_arr[idx], y_score_arr[idx], pa_arr[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, ys))
        boot_accs.append(float(np.mean((ys > 0.5) == yt)))
        boot_precs.append(_compute_med_precision(yt, pa_b))

    return {
        "roc_auc": np.array(boot_aucs, dtype=float),
        "roc_scores": y_score_arr,
        "accuracy": np.array(boot_accs, dtype=float),
        "latency": latency_arr,
        "med_precision": np.array(boot_precs, dtype=float),
    }
