"""
reward_function.py
==================
Equation 1 from paper:
    R_t = R_t^clinical + lambda_adh * R_t^adherence + lambda_safe * R_t^safety
"""

import numpy as np


def compute_reward(
    state: np.ndarray, action: int, vital: dict, config: dict, action_mask: np.ndarray
) -> float:
    sbp = vital.get("sbp", 130)
    glc = vital.get("glucose_mgdl", 100)
    spo2 = vital.get("spo2", 97)
    sbp_lo, sbp_hi = config["bp_target_systolic"]
    glc_lo, glc_hi = config["glucose_tir_target"]

    bp_r = (
        1.0
        if sbp_lo <= sbp <= sbp_hi
        else -abs(sbp - np.clip(sbp, sbp_lo, sbp_hi)) / 20.0
    )
    tir_r = (
        1.0
        if glc_lo <= glc <= glc_hi
        else -abs(glc - np.clip(glc, glc_lo, glc_hi)) / 50.0
    )
    spo2_r = 1.0 if spo2 >= 94 else -abs(spo2 - 94) / 5.0
    r_clinical = config["clinical_stability_weight"] * (bp_r + tir_r + spo2_r) / 3.0

    adh_med = vital.get("adherence_med", 0.7)
    adh_diet = vital.get("adherence_diet", 0.5)
    r_adherence = (adh_med + adh_diet) / 2.0 - 0.5

    r_safety = 0.0
    if not action_mask[action]:
        r_safety += config["constraint_violation_penalty"]
    event = vital.get("event_type")
    if event and action != 4:
        r_safety += config["escalation_event_penalty"]
    elif action == 4 and not event:
        r_safety += config["escalation_event_penalty"] * 0.5

    R_t = (
        r_clinical
        + config["lambda_adherence"] * r_adherence
        + config["lambda_safety"] * r_safety
    )
    return float(R_t)
