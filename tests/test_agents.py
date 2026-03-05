"""Unit tests for all four BDI agents.

Tests use the actual agent API:
    agent.execute(task: dict) → {"agent": str, "actions": list, "metadata": dict}
    agent._safety_gate(action: AgentAction) → bool
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from governance.audit_log      import AuditLog
from knowledge.policy_registry import PolicyRegistry
from knowledge.knowledge_graph import KnowledgeGraph


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def audit_log(tmp_path):
    return AuditLog(str(tmp_path / "agents_test.jsonl"))

@pytest.fixture
def registry():
    return PolicyRegistry()

@pytest.fixture
def kg():
    return KnowledgeGraph()


# ─── Nutrition Agent ──────────────────────────────────────────────────────────

def test_nutrition_agent_returns_correct_structure(audit_log, registry, kg):
    """execute() must return the standard three-key dict."""
    from agents.nutrition_agent import NutritionAgent
    agent  = NutritionAgent(registry, kg, audit_log)
    result = agent.execute({
        "bmi": 28, "conditions": ["hypertension"],
        "vitals": {"glucose_mgdl": 100}, "allergies": {},
        "patient_id": "p001", "prescriptions": [],
    })
    assert "agent"    in result
    assert "actions"  in result
    assert "metadata" in result
    assert result["agent"] == "nutrition_agent"


def test_nutrition_agent_builds_meal_plan(audit_log, registry, kg):
    """execute() must produce at least one MEAL_PLAN action."""
    from agents.nutrition_agent import NutritionAgent
    agent  = NutritionAgent(registry, kg, audit_log)
    result = agent.execute({
        "bmi": 28, "conditions": ["hypertension"],
        "vitals": {"glucose_mgdl": 100}, "allergies": {},
        "patient_id": "p001", "prescriptions": [],
    })
    action_types = [a.get("action_type", "") for a in result["actions"]]
    assert "meal_plan" in action_types, (
        f"Expected meal_plan action; got {action_types}")


def test_nutrition_agent_sodium_below_cap(audit_log, registry, kg):
    """Meal plan sodium must be ≤ 2 300 mg + tolerance for hypertensive patient."""
    from agents.nutrition_agent import NutritionAgent
    agent  = NutritionAgent(registry, kg, audit_log)
    result = agent.execute({
        "bmi": 28, "conditions": ["hypertension"],
        "vitals": {"glucose_mgdl": 100}, "allergies": {},
        "patient_id": "p001", "prescriptions": [],
    })
    # Find the meal_plan action payload
    for action in result["actions"]:
        if action.get("action_type") == "meal_plan":
            sodium = action.get("total_sodium_mg", 0)
            assert sodium <= 2300 + 500, (
                f"Sodium {sodium} mg exceeds 2300 mg cap + 500 mg tolerance")
            return
    pytest.fail("No meal_plan action found in result")


# ─── Lifestyle Agent ──────────────────────────────────────────────────────────

def test_lifestyle_agent_caffeine_warning(audit_log, registry, kg):
    """Caffeine consumed after 14:00 must produce a caffeine_hygiene action."""
    from agents.lifestyle_agent import LifestyleAgent
    agent  = LifestyleAgent(registry, kg, audit_log)
    result = agent.execute({
        "hour_of_day": 20,
        "conditions": ["hypertension"],
        "caffeine_intake_mg": 200,
        "steps_today": 5000,
        "sleep_actual_hours": 6.0,
        "sleep_target_hours": 7.5,
        "vitals": {"heart_rate": 72},
        "patient_id": "p002",
    })
    action_types = [a.get("action_type", "") for a in result["actions"]]
    assert "caffeine_hygiene" in action_types, (
        f"Expected caffeine_hygiene in {action_types}")


def test_lifestyle_agent_sleep_advance_on_debt(audit_log, registry, kg):
    """More than 0.5 h sleep debt must produce a sleep_adjustment action."""
    from agents.lifestyle_agent import LifestyleAgent
    agent  = LifestyleAgent(registry, kg, audit_log)
    result = agent.execute({
        "hour_of_day": 10,
        "conditions": [],
        "caffeine_intake_mg": 0,
        "steps_today": 5000,
        "sleep_actual_hours": 5.0,   # debt = 2.5 h
        "sleep_target_hours": 7.5,
        "vitals": {"heart_rate": 68},
        "patient_id": "p002",
    })
    action_types = [a.get("action_type", "") for a in result["actions"]]
    assert "sleep_adjustment" in action_types, (
        f"Expected sleep_adjustment in {action_types}")


# ─── Medicine Agent ───────────────────────────────────────────────────────────

def test_medicine_agent_safety_gate_blocks_new_medication(audit_log, registry, kg):
    """_safety_gate must block any action whose payload contains add_medication."""
    from agents.medicine_agent import MedicineAgent
    from agents.base_agent import AgentAction, ActionType, Urgency
    agent  = MedicineAgent(registry, kg, audit_log)
    # Action with add_medication key → must be blocked
    bad_action = AgentAction(
        action_type = ActionType.MEDICATION_REMINDER,
        urgency     = Urgency.ROUTINE,
        payload     = {"add_medication": True, "drug": "aspirin"},
        agent_id    = "medicine_agent",
        patient_id  = "p003",
    )
    assert not agent._safety_gate(bad_action), (
        "_safety_gate should return False for add_medication payload")


def test_medicine_agent_safety_gate_permits_reminder(audit_log, registry, kg):
    """_safety_gate must allow a normal medication reminder."""
    from agents.medicine_agent import MedicineAgent
    from agents.base_agent import AgentAction, ActionType, Urgency
    agent  = MedicineAgent(registry, kg, audit_log)
    good_action = AgentAction(
        action_type = ActionType.MEDICATION_REMINDER,
        urgency     = Urgency.ROUTINE,
        payload     = {"drug": "metformin", "dose_mg": 500},
        agent_id    = "medicine_agent",
        patient_id  = "p003",
    )
    assert agent._safety_gate(good_action), (
        "_safety_gate should return True for a safe reminder")


def test_medicine_agent_adherence_reminder(audit_log, registry, kg):
    """Low adherence should trigger a medication_reminder action."""
    from agents.medicine_agent import MedicineAgent
    from agents.base_agent import MedicationEntry
    agent  = MedicineAgent(registry, kg, audit_log)
    result = agent.execute({
        "patient_id":    "p003",
        "prescriptions": [MedicationEntry(
            drug="metformin", dose_mg=500,
            frequency="twice_daily", timing="with_meals")],
        "adherence_med": 0.40,     # below 0.75 threshold
        "labs": None,
        "vitals": {},
        "conditions": [],
        "allergies": {},
        "proposed_foods": [],
    })
    action_types = [a.get("action_type", "") for a in result["actions"]]
    assert "medication_reminder" in action_types, (
        f"Expected medication_reminder for low adherence; got {action_types}")


# ─── Emergency Agent ──────────────────────────────────────────────────────────

def test_emergency_agent_schedules_repeat_on_high_bp(audit_log, registry, kg):
    """First call with SBP ≥ 180 must schedule a repeat measurement."""
    from agents.emergency_agent import EmergencyAgent
    agent  = EmergencyAgent(registry, kg, audit_log)
    result = agent.execute({
        "patient_id": "p004",
        "vitals": {
            "sbp": 185, "dbp": 112, "glucose_mgdl": 100,
            "heart_rate": 88, "spo2": 96,
        },
    })
    status = result["metadata"].get("status", "")
    assert status in ("REPEAT_SCHEDULED", "ESCALATED_TO_CLINICIAN"), (
        f"Unexpected status: {status}")


def test_emergency_agent_no_action_on_normal_vitals(audit_log, registry, kg):
    """Normal vitals must produce NO_ACTION status with zero actions."""
    from agents.emergency_agent import EmergencyAgent
    agent  = EmergencyAgent(registry, kg, audit_log)
    result = agent.execute({
        "patient_id": "p005",
        "vitals": {
            "sbp": 118, "dbp": 76, "glucose_mgdl": 95,
            "heart_rate": 70, "spo2": 98,
        },
    })
    assert result["metadata"]["status"] == "NO_ACTION"
    assert result["actions"] == []


def test_emergency_agent_escalates_after_persistence_threshold(audit_log, registry, kg):
    """After persistence_threshold repeat calls with persistent abnormals, must escalate.

    With persistence_threshold=2 the state machine requires:
      - Call 1: repeat_count 0 → 1  (REPEAT_SCHEDULED)
      - Call 2: repeat_count 1 → 2  (REPEAT_SCHEDULED)
      - Call 3: repeat_count == threshold, still abnormal → ESCALATED_TO_CLINICIAN
    """
    from agents.emergency_agent import EmergencyAgent
    agent = EmergencyAgent(registry, kg, audit_log, persistence_threshold=2)
    hypertensive_vitals = {
        "patient_id": "p006",
        "vitals": {
            "sbp": 188, "dbp": 114, "glucose_mgdl": 100,
            "heart_rate": 90, "spo2": 96,
        },
    }
    r1 = agent.execute(hypertensive_vitals)
    assert r1["metadata"]["status"] == "REPEAT_SCHEDULED", (
        f"Call 1: expected REPEAT_SCHEDULED, got {r1['metadata']['status']}")

    r2 = agent.execute(hypertensive_vitals)
    assert r2["metadata"]["status"] == "REPEAT_SCHEDULED", (
        f"Call 2: expected REPEAT_SCHEDULED, got {r2['metadata']['status']}")

    # Third call: count reaches threshold → must escalate if vitals still abnormal
    r3 = agent.execute(hypertensive_vitals)
    assert r3["metadata"]["status"] == "ESCALATED_TO_CLINICIAN", (
        f"Call 3: expected ESCALATED_TO_CLINICIAN after {2} repeats; "
        f"got {r3['metadata']['status']}")
