"""
mimic_evaluation.py
===================
Runs Algorithm 1 (anomaly detection) on MIMIC-IV real patient data.
Reports ROC AUC vs ICD-coded escalation ground truth.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix


def evaluate_mimic(vitals_path: str, events_path: str) -> dict:
    vitals_df = pd.read_csv(vitals_path)
    events_df = pd.read_csv(events_path)

    # Build ground truth: 1 if patient had any escalation event
    patients_with_event = set(events_df["subject_id"].astype(str).unique())

    # Map MIMIC column names to our channel names
    col_map = {
        "systolic_bp": "sbp",
        "diastolic_bp": "dbp",
        "heart_rate": "heart_rate",
        "spo2": "spo2",
        "glucose": "glucose_mgdl",
    }
    for mimic_col, our_col in col_map.items():
        if mimic_col in vitals_df.columns and our_col not in vitals_df.columns:
            vitals_df[our_col] = vitals_df[mimic_col]

    # Fill required columns
    for col in ["sbp", "dbp", "glucose_mgdl", "heart_rate", "spo2"]:
        if col not in vitals_df.columns:
            vitals_df[col] = np.nan

    # Apply escalation thresholds (Algorithm 1)
    import yaml

    with open("configs/escalation_thresholds.yaml") as f:
        thr = yaml.safe_load(f)
    bp = thr["blood_pressure"]
    glc = thr["glucose"]
    spo2_thr = thr["spo2"]

    def anomaly_score(row) -> float:
        score = 0.1
        sbp = row.get("sbp") or 120
        glucose = row.get("glucose_mgdl") or 100
        spo2 = row.get("spo2") or 97
        if sbp >= bp["systolic_emergency"]:
            score = 0.95
        elif sbp >= bp["systolic_urgent"]:
            score = max(score, 0.70)
        if glucose <= glc["hypoglycemia_severe"]:
            score = 0.95
        elif glucose <= glc["hypoglycemia_mild"]:
            score = max(score, 0.65)
        if spo2 <= spo2_thr["emergency"]:
            score = 0.95
        elif spo2 <= spo2_thr["alert"]:
            score = max(score, 0.60)
        return float(np.clip(score + np.random.normal(0, 0.03), 0, 1))

    patient_col = "stay_id" if "stay_id" in vitals_df.columns else "subject_id"
    patient_ids = vitals_df[patient_col].astype(str).unique()

    y_true, y_score_arr = [], []
    for pid in patient_ids:
        grp = vitals_df[vitals_df[patient_col].astype(str) == pid]
        max_score = grp.apply(anomaly_score, axis=1).max()
        label = 1 if pid in patients_with_event else 0
        y_true.append(label)
        y_score_arr.append(max_score)

    y_true = np.array(y_true)
    y_score = np.array(y_score_arr)

    if len(np.unique(y_true)) < 2:
        return {
            "error": "Not enough positive examples in MIMIC subset",
            "n_patients": len(patient_ids),
        }

    roc = roc_auc_score(y_true, y_score)
    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) >= 2:
            boot.append(roc_auc_score(y_true[idx], y_score[idx]))

    tn, fp, fn, tp = confusion_matrix(y_true, y_score > 0.5).ravel()

    return {
        "n_patients": int(len(patient_ids)),
        "n_positive": int(y_true.sum()),
        "roc_auc": float(roc),
        "auc_ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        "sensitivity": float(tp / (tp + fn + 1e-8)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "fpr": float(fp / (fp + tn + 1e-8)),
    }
