"""Unit tests for the Constraint Filter (safety layer)."""
import sys, os, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knowledge.policy_registry import PolicyRegistry
from orchestrator.constraint_filter import ConstraintFilter


@pytest.fixture
def cf():
    return ConstraintFilter(PolicyRegistry())


def test_mandatory_escalation_override(cf):
    """When BP >= 180, action MUST be escalate regardless of proposed action."""
    vitals = {"sbp": 185, "dbp": 110, "glucose_mgdl": 100,
               "heart_rate": 80, "spo2": 96}
    filtered, reason = cf.filter(0, vitals, 0.7, {})  # proposed: no_action
    assert filtered == 4, "Should override to escalate"


def test_no_med_reminder_high_adherence(cf):
    vitals = {"sbp": 120, "dbp": 78, "glucose_mgdl": 100,
               "heart_rate": 72, "spo2": 97}
    filtered, reason = cf.filter(1, vitals, 0.95, {})
    assert filtered == 0, "Should not send med reminder when adherence=0.95"


def test_action_mask_blocks_unnecessary_escalation(cf):
    vitals = {"sbp": 118, "dbp": 76, "glucose_mgdl": 95,
               "heart_rate": 70, "spo2": 98}
    mask = cf.action_mask(vitals, 0.75)
    assert mask[4] == False, "Escalation should be masked for normal vitals"
    assert mask[0] == True, "No-action always available"
