"""Unit tests for composite reward (Equation 1)."""
import sys, os, numpy as np, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.reward_function import compute_reward

CFG = {
    "lambda_adherence": 0.3, "lambda_safety": 2.0,
    "clinical_stability_weight": 1.0,
    "bp_target_systolic": [120, 130],
    "glucose_tir_target": [70, 180],
    "constraint_violation_penalty": -10.0,
    "escalation_event_penalty": -5.0,
}

def test_reward_in_range():
    vital = {"sbp": 125, "dbp": 80, "glucose_mgdl": 110,
              "spo2": 97, "adherence_med": 0.8, "adherence_diet": 0.6}
    mask = np.ones(5, dtype=bool)
    r = compute_reward(np.zeros(25), 0, vital, CFG, mask)
    assert -20 < r < 5

def test_escalation_missed_penalty():
    vital = {"sbp": 130, "glucose_mgdl": 50, "spo2": 97,
              "adherence_med": 0.7, "event_type": "hypoglycemic_episode"}
    mask = np.ones(5, dtype=bool)
    r_miss = compute_reward(np.zeros(25), 0, vital, CFG, mask)
    r_esc  = compute_reward(np.zeros(25), 4, vital, CFG, mask)
    assert r_esc > r_miss

def test_constraint_violation_penalty():
    vital = {"sbp": 120, "glucose_mgdl": 100, "spo2": 97,
              "adherence_med": 0.7}
    mask_ok  = np.ones(5, dtype=bool)
    mask_bad = np.ones(5, dtype=bool); mask_bad[2] = False  # action 2 blocked
    r_ok  = compute_reward(np.zeros(25), 2, vital, CFG, mask_ok)
    r_bad = compute_reward(np.zeros(25), 2, vital, CFG, mask_bad)
    assert r_bad < r_ok
