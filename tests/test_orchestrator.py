"""tests/test_orchestrator.py — Integration tests for the Orchestrator.

These tests exercise the ACTUAL Orchestrator API:
    __init__(feature_store, knowledge_graph, policy_registry, agents, audit_log, config_path)
    step(patient_id, raw_vitals, patient_state, observation=None)
      → {"patient_id", "goals", "events", "tasks_resolved", "results", "context_summary"}
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from governance.audit_log import AuditLog
from knowledge.policy_registry import PolicyRegistry
from knowledge.knowledge_graph import KnowledgeGraph
from knowledge.feature_store import FeatureStore
from agents.medicine_agent import MedicineAgent
from agents.nutrition_agent import NutritionAgent
from agents.lifestyle_agent import LifestyleAgent
from agents.emergency_agent import EmergencyAgent

# ─── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_audit(tmp_path):
    return AuditLog(str(tmp_path / "audit_test.jsonl"))


@pytest.fixture
def registry():
    return PolicyRegistry()


@pytest.fixture
def kg():
    return KnowledgeGraph()


@pytest.fixture
def feature_store():
    return FeatureStore()


@pytest.fixture
def agents(registry, kg, tmp_audit):
    return {
        "medicine_agent": MedicineAgent(registry, kg, tmp_audit),
        "nutrition_agent": NutritionAgent(registry, kg, tmp_audit),
        "lifestyle_agent": LifestyleAgent(registry, kg, tmp_audit),
        "emergency_agent": EmergencyAgent(registry, kg, tmp_audit),
    }


@pytest.fixture
def orchestrator(feature_store, kg, registry, agents, tmp_audit):
    from orchestrator.orchestrator import Orchestrator

    return Orchestrator(
        feature_store=feature_store,
        knowledge_graph=kg,
        policy_registry=registry,
        agents=agents,
        audit_log=tmp_audit,
        config_path="configs/rl_training.yaml",
    )


def _patient_state():
    """Minimal patient state dict accepted by orchestrator.step()."""
    return {
        "patient_id": 42,
        "conditions": ["hypertension", "type2_diabetes"],
        "prescriptions": [
            {
                "drug": "metformin",
                "dose_mg": 500,
                "frequency": "twice_daily",
                "timing": "with_meals",
            }
        ],
        "labs": {"egfr": 75, "ast": 22, "alt": 20},
    }


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_orchestrator_step_returns_required_keys(orchestrator):
    """step() must return the six documented top-level keys."""
    vitals = {
        "sbp": 130,
        "dbp": 82,
        "glucose_mgdl": 140,
        "heart_rate": 72,
        "spo2": 97,
        "adherence_med": 0.75,
        "adherence_diet": 0.55,
    }
    result = orchestrator.step(42, vitals, _patient_state())

    for key in (
        "patient_id",
        "goals",
        "events",
        "tasks_resolved",
        "results",
        "context_summary",
    ):
        assert key in result, f"Missing key: '{key}'"


def test_orchestrator_goals_is_list(orchestrator):
    """goals must be a non-empty list of strings."""
    vitals = {
        "sbp": 130,
        "dbp": 82,
        "glucose_mgdl": 140,
        "heart_rate": 72,
        "spo2": 97,
        "adherence_med": 0.75,
        "adherence_diet": 0.55,
    }
    result = orchestrator.step(42, vitals, _patient_state())
    assert isinstance(result["goals"], list)
    assert len(result["goals"]) >= 1


def test_orchestrator_emergency_triggered_on_high_bp(orchestrator):
    """SBP ≥ 180 should produce an acute_vital_exceedance event and
    an emergency_escalation goal."""
    vitals = {
        "sbp": 192,
        "dbp": 112,
        "glucose_mgdl": 120,
        "heart_rate": 80,
        "spo2": 97,
        "adherence_med": 0.8,
        "adherence_diet": 0.6,
    }
    result = orchestrator.step(42, vitals, _patient_state())

    event_types = [ev["type"] for ev in result["events"]]
    assert (
        "acute_vital_exceedance" in event_types
    ), f"Expected acute_vital_exceedance in {event_types}"

    assert (
        "emergency_escalation" in result["goals"]
    ), f"Expected emergency_escalation in {result['goals']}"


def test_orchestrator_audit_logged(orchestrator, tmp_audit):
    """Each orchestration cycle must write at least one audit entry."""
    initial_count = tmp_audit.entry_count
    vitals = {
        "sbp": 125,
        "dbp": 80,
        "glucose_mgdl": 110,
        "heart_rate": 70,
        "spo2": 98,
        "adherence_med": 0.9,
        "adherence_diet": 0.7,
    }
    orchestrator.step(42, vitals, _patient_state())
    assert (
        tmp_audit.entry_count > initial_count
    ), "Audit log must grow after an orchestration cycle"


def test_orchestrator_tasks_resolved_count(orchestrator):
    """tasks_resolved must be a non-negative integer."""
    vitals = {
        "sbp": 125,
        "dbp": 80,
        "glucose_mgdl": 110,
        "heart_rate": 70,
        "spo2": 98,
        "adherence_med": 0.9,
        "adherence_diet": 0.7,
    }
    result = orchestrator.step(42, vitals, _patient_state())
    assert isinstance(result["tasks_resolved"], int)
    assert result["tasks_resolved"] >= 0


def test_orchestrator_sodium_cap_in_context(orchestrator):
    """context_summary must report the sodium cap used."""
    vitals = {"sbp": 125, "dbp": 80, "glucose_mgdl": 110, "heart_rate": 70, "spo2": 98}
    result = orchestrator.step(42, vitals, _patient_state())
    assert "sodium_cap" in result["context_summary"]
    # Hypertensive patient → 2.3 g/day
    assert result["context_summary"]["sodium_cap"] == pytest.approx(2.3, abs=0.1)
