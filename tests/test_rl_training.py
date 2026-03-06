"""Unit tests for RL training modules (train.py, lagrangian.py, evaluate_policy.py)."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.patient_env import PatientEnv
from rl.evaluate_policy import (
    _build_channel_statistics,
    _build_observation,
    _load_channel_statistics,
    _predict,
    _build_action_mask,
    _compute_med_precision,
    _compute_latency_seconds,
)
from rl.lagrangian import LagrangianUpdater
from knowledge.policy_registry import PolicyRegistry


class TestChannelStatisticsLoading:
    """Test loading of channel normalization statistics from config."""

    def test_load_channel_statistics_defaults(self):
        """Test that default channel statistics are loaded correctly."""
        means, stds = _load_channel_statistics()
        
        assert means.shape == (5,), "Means should have shape (5,)"
        assert stds.shape == (5,), "Stds should have shape (5,)"
        assert means.dtype == np.float32
        assert stds.dtype == np.float32
        
        # Check default values
        expected_means = np.array([130.0, 82.0, 110.0, 72.0, 97.5], dtype=np.float32)
        expected_stds = np.array([20.0, 12.0, 40.0, 12.0, 2.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(means, expected_means)
        np.testing.assert_array_almost_equal(stds, expected_stds)

    def test_channel_statistics_are_positive(self):
        """Test that loaded statistics are strictly positive."""
        means, stds = _load_channel_statistics()
        
        assert np.all(means > 0), "All means must be positive"
        assert np.all(stds > 0), "All stds must be positive"


class TestObservationBuilding:
    """Test construction of 25-dimensional state vectors."""

    def test_observation_shape(self):
        """Test that observation has correct shape and dtype."""
        row = {
            "sbp": 130.0,
            "dbp": 82.0,
            "glucose_mgdl": 110.0,
            "heart_rate": 72.0,
            "spo2": 97.5,
            "adherence_med": 0.7,
            "adherence_diet": 0.5,
            "adherence_lifestyle": 0.6,
            "t_minutes": 100.0,
        }
        
        import pandas as pd
        row_series = pd.Series(row)
        channel_means = np.array([130.0, 82.0, 110.0, 72.0, 97.5], dtype=np.float32)
        channel_stds = np.array([20.0, 12.0, 40.0, 12.0, 2.0], dtype=np.float32)
        
        obs = _build_observation(row_series, [], channel_means, channel_stds)
        
        assert obs.shape == (25,), f"Expected shape (25,), got {obs.shape}"
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs)), "All observation values must be finite"

    def test_observation_bounded(self):
        """Test that observation values are clipped to [-10, 10]."""
        row = {
            "sbp": 300.0,  # Extreme value
            "dbp": 10.0,   # Extreme value
            "glucose_mgdl": 600.0,
            "heart_rate": 20.0,
            "spo2": 50.0,
            "adherence_med": 0.7,
            "adherence_diet": 0.5,
            "adherence_lifestyle": 0.6,
            "t_minutes": 100.0,
        }
        
        import pandas as pd
        row_series = pd.Series(row)
        channel_means = np.array([130.0, 82.0, 110.0, 72.0, 97.5], dtype=np.float32)
        channel_stds = np.array([20.0, 12.0, 40.0, 12.0, 2.0], dtype=np.float32)
        
        obs = _build_observation(row_series, [], channel_means, channel_stds)
        
        assert np.all(obs >= -10.0), f"Observation min: {obs.min()}"
        assert np.all(obs <= 10.0), f"Observation max: {obs.max()}"

    def test_observation_with_window(self):
        """Test observation construction with rolling window."""
        current_row = {
            "sbp": 130.0,
            "dbp": 82.0,
            "glucose_mgdl": 110.0,
            "heart_rate": 72.0,
            "spo2": 97.5,
            "adherence_med": 0.7,
            "adherence_diet": 0.5,
            "adherence_lifestyle": 0.6,
            "t_minutes": 100.0,
        }
        
        window = [
            {"sbp": 128.0, "dbp": 81.0, "glucose_mgdl": 105.0, "heart_rate": 70.0, "spo2": 98.0},
            {"sbp": 132.0, "dbp": 83.0, "glucose_mgdl": 115.0, "heart_rate": 74.0, "spo2": 97.0},
        ]
        
        import pandas as pd
        row_series = pd.Series(current_row)
        channel_means = np.array([130.0, 82.0, 110.0, 72.0, 97.5], dtype=np.float32)
        channel_stds = np.array([20.0, 12.0, 40.0, 12.0, 2.0], dtype=np.float32)
        
        obs = _build_observation(row_series, window, channel_means, channel_stds)
        
        assert obs.shape == (25,)
        assert np.all(np.isfinite(obs))


class TestActionMasking:
    """Test action mask generation for constraint satisfaction."""

    def test_action_mask_shape(self):
        """Test that action mask has correct shape."""
        registry = PolicyRegistry()
        vitals = {"sbp": 120.0, "glucose_mgdl": 100.0}
        mask = _build_action_mask(vitals, registry)
        
        assert mask.shape == (5,), f"Expected shape (5,), got {mask.shape}"
        assert mask.dtype == bool

    def test_escalation_always_available_in_crisis(self):
        """Test that escalation action is available during crisis."""
        registry = PolicyRegistry()
        # High SBP triggers escalation
        vitals = {"sbp": 220.0, "glucose_mgdl": 100.0, "spo2": 95.0, "heart_rate": 80.0}
        mask = _build_action_mask(vitals, registry)
        
        # Action 4 is escalation
        assert mask[4] == True, "Escalation should be available in crisis"

    def test_no_action_always_available(self):
        """Test that no-action is always available."""
        registry = PolicyRegistry()
        vitals = {"sbp": 120.0, "glucose_mgdl": 100.0, "adherence_med": 0.5}
        mask = _build_action_mask(vitals, registry)
        
        # Action 0 is no-action
        assert mask[0] == True, "No-action should always be available"


class TestMedicationPrecision:
    """Test medication recommendation precision metric."""

    def test_med_precision_perfect(self):
        """Test precision when all escalations are correct."""
        y_true = np.array([0, 1, 1, 0, 1])
        policy_actions = np.array([0, 4, 4, 0, 4])  # Escalates (action 4) on true events
        
        prec = _compute_med_precision(y_true, policy_actions)
        
        assert prec == 1.0, "Perfect escalations should have precision 1.0"

    def test_med_precision_zero_division(self):
        """Test precision when policy never escalates."""
        y_true = np.array([0, 1, 1, 0, 1])
        policy_actions = np.array([0, 0, 0, 0, 0])  # Never escalates
        
        prec = _compute_med_precision(y_true, policy_actions)
        
        assert prec == 0.0, "Policy that never escalates should have precision 0"

    def test_med_precision_partial(self):
        """Test precision with mixed escalations."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        policy_actions = np.array([0, 4, 4, 4, 4, 0])  # 1 FP (index 3)
        
        # 3 TP, 1 FP -> precision = 3/4 = 0.75
        prec = _compute_med_precision(y_true, policy_actions)
        
        assert abs(prec - 0.75) < 0.01, f"Expected ~0.75, got {prec}"


class TestLatencyComputation:
    """Test latency computation from event to escalation."""

    def test_latency_basic(self):
        """Test latency computation for a simple case."""
        import pandas as pd
        
        vitals_df = pd.DataFrame({
            "patient_id": [1, 1, 1, 1, 1],
            "t_minutes": [0, 5, 10, 15, 20],
            "sbp": [120, 120, 220, 220, 220],  # High at t=10
            "glucose_mgdl": [100, 100, 100, 100, 100],
            "spo2": [98, 98, 98, 98, 98],
            "heart_rate": [70, 70, 70, 70, 70],
        })
        
        event_set = {(1, 10)}  # Event at patient 1, t=10 min
        policy_actions_map = {(1, 0): 0, (1, 5): 0, (1, 10): 0, (1, 15): 4, (1, 20): 4}
        registry = PolicyRegistry()
        
        latencies = _compute_latency_seconds(vitals_df, event_set, policy_actions_map, registry)
        
        # Event at t=10, escalation at t=15 -> 5 min = 300 sec
        assert len(latencies) > 0, "Should have computed at least one latency"
        assert all(0 <= lat <= 3600 for lat in latencies), "Latencies should be in [0, 3600]"


class TestLagrangianUpdater:
    """Test Lagrangian multiplier updates for constraint satisfaction."""

    def test_lagrangian_initialization(self):
        """Test that Lagrangian updater initializes correctly."""
        updater = LagrangianUpdater(
            constraint_threshold=0.05,
            lagrangian_lr=0.01,
            lambda_init=1.0,
            lambda_max=10.0,
        )
        
        assert updater.lambda_current == 1.0, "Initial lambda should be 1.0"
        assert updater.constraint_threshold == 0.05
        assert updater.lagrangian_lr == 0.01

    def test_lagrangian_increase_on_violation(self):
        """Test that lambda increases when constraint is violated."""
        updater = LagrangianUpdater(
            constraint_threshold=0.05,
            lagrangian_lr=0.01,
            lambda_init=1.0,
            lambda_max=10.0,
        )
        
        initial_lambda = updater.lambda_current
        # Constraint violation cost > threshold
        violation_cost = 0.10
        
        updater.update(violation_cost)
        
        assert updater.lambda_current > initial_lambda, "Lambda should increase on constraint violation"

    def test_lagrangian_decrease_on_satisfaction(self):
        """Test that lambda decreases when constraint is satisfied."""
        updater = LagrangianUpdater(
            constraint_threshold=0.05,
            lagrangian_lr=0.01,
            lambda_init=5.0,
            lambda_max=10.0,
        )
        
        initial_lambda = updater.lambda_current
        # Constraint satisfied (cost < threshold)
        violation_cost = 0.01
        
        updater.update(violation_cost)
        
        assert updater.lambda_current < initial_lambda, "Lambda should decrease when constraint is satisfied"

    def test_lagrangian_bounded(self):
        """Test that lambda respects bounds."""
        updater = LagrangianUpdater(
            constraint_threshold=0.05,
            lagrangian_lr=10.0,  # Large LR for testing
            lambda_init=1.0,
            lambda_max=5.0,
        )
        
        # Many violations to test upper bound
        for _ in range(100):
            updater.update(0.50)
        
        assert updater.lambda_current <= 5.0, "Lambda should not exceed lambda_max"
        assert updater.lambda_current >= 0.0, "Lambda should be non-negative"


class TestPatientEnvironmentIntegration:
    """Integration tests with PatientEnv."""

    @pytest.mark.slow
    def test_patient_env_creates_correct_obs(self):
        """Test that PatientEnv produces observations matching our building logic."""
        env = PatientEnv()
        obs, info = env.reset()
        
        assert obs.shape == (25,), f"Expected obs shape (25,), got {obs.shape}"
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        assert np.all(obs >= -10.0) and np.all(obs <= 10.0)

    @pytest.mark.slow
    def test_env_action_mask_consistency(self):
        """Test that environment action masks are consistent."""
        env = PatientEnv()
        env.reset()
        
        mask = env.action_masks()
        
        assert mask.shape == (5,)
        assert mask.dtype == bool
        assert mask[0] == True, "No-action should always be available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
