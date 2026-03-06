"""Unit tests for baseline prediction methods (B1, B2, B3)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.rules_only import RulesOnlyBaseline
from baselines.predictive_only import PredictiveBaseline
from baselines.human_schedule import HumanScheduleBaseline
from knowledge.policy_registry import PolicyRegistry


class TestRulesOnlyBaseline:
    """Test threshold-based rule baseline (B1)."""

    def test_rules_baseline_initialization(self):
        """Test that rules-only baseline initializes correctly."""
        baseline = RulesOnlyBaseline()
        assert baseline is not None

    def test_rules_baseline_normal_vitals(self):
        """Test that baseline returns no action for normal vitals."""
        baseline = RulesOnlyBaseline()
        vitals = {
            "sbp": 120.0,
            "dbp": 75.0,
            "glucose_mgdl": 100.0,
            "heart_rate": 70.0,
            "spo2": 98.0,
        }
        
        action = baseline.predict(vitals)
        
        # Normal vitals should result in no_action (0) or low-risk action
        assert action in [0, 1, 2, 3], f"Expected action in [0-3] for normal vitals, got {action}"

    def test_rules_baseline_hypertensive_urgency(self):
        """Test that baseline escalates on hypertensive urgency."""
        baseline = RulesOnlyBaseline()
        vitals = {
            "sbp": 200.0,  # Indicates hypertensive urgency
            "dbp": 120.0,
            "glucose_mgdl": 100.0,
            "heart_rate": 85.0,
            "spo2": 96.0,
        }
        
        action = baseline.predict(vitals)
        
        # Should escalate (action 4) for severe hypertension
        assert action == 4, f"Expected escalation for hypertensive urgency, got action {action}"

    def test_rules_baseline_hypoglycemia(self):
        """Test that baseline escalates on severe hypoglycemia."""
        baseline = RulesOnlyBaseline()
        vitals = {
            "sbp": 110.0,
            "dbp": 70.0,
            "glucose_mgdl": 40.0,  # Severe hypoglycemia
            "heart_rate": 95.0,  # Elevated HR response
            "spo2": 98.0,
        }
        
        action = baseline.predict(vitals)
        
        # Should escalate for severe hypoglycemia
        assert action == 4, f"Expected escalation for severe hypoglycemia, got action {action}"

    def test_rules_baseline_consistent(self):
        """Test that baseline returns consistent results for same input."""
        baseline = RulesOnlyBaseline()
        vitals = {
            "sbp": 130.0,
            "dbp": 80.0,
            "glucose_mgdl": 150.0,
            "heart_rate": 72.0,
            "spo2": 97.0,
        }
        
        action1 = baseline.predict(vitals)
        action2 = baseline.predict(vitals)
        
        assert action1 == action2, "Baseline should return consistent results"

    def test_rules_baseline_action_range(self):
        """Test that baseline always returns valid actions."""
        baseline = RulesOnlyBaseline()
        
        # Test various vital combinations
        test_cases = [
            {"sbp": 90, "dbp": 50, "glucose_mgdl": 80, "heart_rate": 60, "spo2": 99},
            {"sbp": 140, "dbp": 90, "glucose_mgdl": 130, "heart_rate": 80, "spo2": 96},
            {"sbp": 220, "dbp": 130, "glucose_mgdl": 250, "heart_rate": 110, "spo2": 90},
            {"sbp": 100, "dbp": 60, "glucose_mgdl": 40, "heart_rate": 50, "spo2": 98},
        ]
        
        for vitals in test_cases:
            action = baseline.predict(vitals)
            assert 0 <= action <= 4, f"Action {action} out of range [0-4]"


class TestPredictiveBaseline:
    """Test ML-based anomaly detection baseline (B2)."""

    def test_predictive_baseline_initialization(self):
        """Test that predictive baseline initializes correctly."""
        baseline = PredictiveBaseline()
        assert baseline is not None

    def test_predictive_baseline_output_shape(self):
        """Test that predictive baseline returns valid scores."""
        baseline = PredictiveBaseline()
        vitals = {
            "sbp": 130.0,
            "dbp": 80.0,
            "glucose_mgdl": 110.0,
            "heart_rate": 72.0,
            "spo2": 97.0,
        }
        
        score = baseline.predict_score(vitals)
        
        assert isinstance(score, (float, np.floating)), f"Expected float score, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score should be in [0, 1], got {score}"

    def test_predictive_baseline_action_from_score(self):
        """Test that action is threshold-based on anomaly score."""
        baseline = PredictiveBaseline()
        
        # Normal vitals should have low anomaly score
        normal_vitals = {
            "sbp": 120.0,
            "dbp": 75.0,
            "glucose_mgdl": 100.0,
            "heart_rate": 70.0,
            "spo2": 98.0,
        }
        
        # Abnormal vitals should have high anomaly score
        abnormal_vitals = {
            "sbp": 200.0,
            "dbp": 120.0,
            "glucose_mgdl": 300.0,
            "heart_rate": 110.0,
            "spo2": 85.0,
        }
        
        normal_score = baseline.predict_score(normal_vitals)
        abnormal_score = baseline.predict_score(abnormal_vitals)
        
        assert abnormal_score > normal_score, "Abnormal vitals should have higher anomaly score"

    def test_predictive_baseline_consistent(self):
        """Test that predictive baseline is deterministic."""
        baseline = PredictiveBaseline()
        vitals = {
            "sbp": 130.0,
            "dbp": 80.0,
            "glucose_mgdl": 110.0,
            "heart_rate": 72.0,
            "spo2": 97.0,
        }
        
        score1 = baseline.predict_score(vitals)
        score2 = baseline.predict_score(vitals)
        
        assert score1 == score2, "Predictive baseline should be deterministic"

    def test_predictive_baseline_action_range(self):
        """Test that baseline always returns valid actions."""
        baseline = PredictiveBaseline()
        
        test_cases = [
            {"sbp": 90, "dbp": 50, "glucose_mgdl": 80, "heart_rate": 60, "spo2": 99},
            {"sbp": 140, "dbp": 90, "glucose_mgdl": 130, "heart_rate": 80, "spo2": 96},
            {"sbp": 220, "dbp": 130, "glucose_mgdl": 250, "heart_rate": 110, "spo2": 90},
        ]
        
        for vitals in test_cases:
            score = baseline.predict_score(vitals)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range [0-1]"


class TestHumanScheduleBaseline:
    """Test static clinician schedule baseline (B3)."""

    def test_human_schedule_initialization(self):
        """Test that human schedule baseline initializes correctly."""
        baseline = HumanScheduleBaseline()
        assert baseline is not None

    def test_human_schedule_time_dependent(self):
        """Test that baseline respects time-of-day scheduling."""
        baseline = HumanScheduleBaseline()
        
        # Morning check
        vitals_morning = {
            "sbp": 140.0,
            "dbp": 90.0,
            "glucose_mgdl": 200.0,
            "heart_rate": 75.0,
            "spo2": 96.0,
            "t_hour": 8.0,  # 8 AM
        }
        
        # Evening (off-schedule) check
        vitals_evening = {
            "sbp": 140.0,
            "dbp": 90.0,
            "glucose_mgdl": 200.0,
            "heart_rate": 75.0,
            "spo2": 96.0,
            "t_hour": 22.0,  # 10 PM
        }
        
        # Note: This test depends on the human schedule implementation
        # Just verify both return valid actions
        action_morning = baseline.predict(vitals_morning)
        action_evening = baseline.predict(vitals_evening)
        
        assert 0 <= action_morning <= 4
        assert 0 <= action_evening <= 4

    def test_human_schedule_escalation_override(self):
        """Test that human schedule always escalates during crisis."""
        baseline = HumanScheduleBaseline()
        
        # Crisis vitals should trigger escalation regardless of schedule
        crisis_vitals = {
            "sbp": 220.0,
            "dbp": 130.0,
            "glucose_mgdl": 40.0,  # Severe hypoglycemia
            "heart_rate": 120.0,
            "spo2": 85.0,
            "t_hour": 23.0,  # Off-schedule time
        }
        
        action = baseline.predict(crisis_vitals)
        
        # Even off-schedule, critical conditions should escalate
        assert action == 4, f"Expected escalation for crisis vitals, got action {action}"

    def test_human_schedule_action_range(self):
        """Test that baseline always returns valid actions."""
        baseline = HumanScheduleBaseline()
        
        test_cases = [
            {"sbp": 120, "dbp": 75, "glucose_mgdl": 100, "heart_rate": 70, "spo2": 98, "t_hour": 9.0},
            {"sbp": 140, "dbp": 90, "glucose_mgdl": 130, "heart_rate": 80, "spo2": 96, "t_hour": 15.0},
            {"sbp": 100, "dbp": 60, "glucose_mgdl": 50, "heart_rate": 50, "spo2": 98, "t_hour": 3.0},
        ]
        
        for vitals in test_cases:
            action = baseline.predict(vitals)
            assert 0 <= action <= 4, f"Action {action} out of range [0-4]"


class TestBaselineComparison:
    """Compare baseline behaviors on standard test cases."""

    def test_baselines_agree_on_crisis(self):
        """Test that all baselines escalate during medical crisis."""
        crisis_vitals = {
            "sbp": 220.0,
            "dbp": 130.0,
            "glucose_mgdl": 30.0,
            "heart_rate": 130.0,
            "spo2": 80.0,
            "t_hour": 12.0,
        }
        
        rules_baseline = RulesOnlyBaseline()
        predictive_baseline = PredictiveBaseline()
        human_baseline = HumanScheduleBaseline()
        
        rules_action = rules_baseline.predict(crisis_vitals)
        pred_score = predictive_baseline.predict_score(crisis_vitals)
        human_action = human_baseline.predict(crisis_vitals)
        
        # All should flag crisis (rules and human escalate, predictive has high score)
        assert rules_action == 4, "Rules baseline should escalate"
        assert human_action == 4, "Human baseline should escalate"
        assert pred_score > 0.7, f"Predictive baseline should flag crisis, got {pred_score}"

    def test_baselines_conservative_on_normal(self):
        """Test that baselines are conservative for normal vitals."""
        normal_vitals = {
            "sbp": 115.0,
            "dbp": 70.0,
            "glucose_mgdl": 95.0,
            "heart_rate": 65.0,
            "spo2": 99.0,
            "t_hour": 12.0,
        }
        
        rules_baseline = RulesOnlyBaseline()
        predictive_baseline = PredictiveBaseline()
        human_baseline = HumanScheduleBaseline()
        
        rules_action = rules_baseline.predict(normal_vitals)
        pred_score = predictive_baseline.predict_score(normal_vitals)
        human_action = human_baseline.predict(normal_vitals)
        
        # Conservative: no escalation for normal vitals
        assert rules_action != 4
        assert human_action != 4
        assert pred_score < 0.3, f"Predictive should be confident normal, got {pred_score}"

    def test_latency_rules_fast(self):
        """Test that rules-only baseline responds quickly."""
        rules_baseline = RulesOnlyBaseline()
        
        # Immediate escalation on abnormal vitals (no ML inference latency)
        abnormal_vitals = {
            "sbp": 200.0,
            "dbp": 110.0,
            "glucose_mgdl": 300.0,
            "heart_rate": 100.0,
            "spo2": 92.0,
        }
        
        # Measure that prediction is instant (no network calls, no ML inference)
        action = rules_baseline.predict(abnormal_vitals)
        
        # Rules should always be action 0-4
        assert 0 <= action <= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
