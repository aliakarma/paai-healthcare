"""
orchestrator.py
===============
Algorithm 2 / Listing 1 from the paper.
The Orchestrator coordinates the BDI agents, applies the RL policy,
enforces the Constraint Filter, resolves cross-domain conflicts,
and routes tasks to specialised agents.

This is the central reasoning hub of the AgHealth+ system.
"""
import numpy as np
import yaml
from typing import Optional

from knowledge.knowledge_graph import KnowledgeGraph
from knowledge.feature_store import FeatureStore
from knowledge.policy_registry import PolicyRegistry
from orchestrator.constraint_filter import ConstraintFilter
from orchestrator.conflict_resolver import ConflictResolver
from orchestrator.task_router import TaskRouter
from governance.audit_log import AuditLog


class Orchestrator:
    """
    Belief-Desire-Intention Orchestrator implementing Listing 1 / Algorithm 2.

    Beliefs  : F (feature store) + context from KG and policies
    Desires  : G (formulated goals)
    Intentions: T' (conflict-resolved, constraint-filtered task list)
    """

    ACTION_NAMES = {
        0: "no_action",
        1: "medication_schedule",
        2: "dietary_modification",
        3: "lifestyle_prompt",
        4: "escalate",
    }

    def __init__(self, feature_store: FeatureStore,
                 knowledge_graph: KnowledgeGraph,
                 policy_registry: PolicyRegistry,
                 agents: dict,
                 audit_log: AuditLog,
                 rl_policy=None,
                 config_path: str = "configs/rl_training.yaml"):
        self.F = feature_store
        self.K = knowledge_graph
        self.P = policy_registry
        self.agents = agents
        self.audit_log = audit_log
        self.rl_policy = rl_policy

        self.constraint_filter = ConstraintFilter(policy_registry)
        self.conflict_resolver = ConflictResolver(knowledge_graph, policy_registry)
        self.task_router = TaskRouter()

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Override log — stores Tier 2 clinician override tuples
        self._override_log: list[dict] = []

    # ------------------------------------------------------------------
    # Step 1: Pull latest features (Listing 1, line 1)
    # ------------------------------------------------------------------
    def pull_latest(self, patient_id: int, raw_vitals: dict) -> np.ndarray:
        self.F.push(patient_id, raw_vitals)
        sbp  = self.F.get_rolling_mean(patient_id, "sbp", 12)
        glc  = self.F.get_rolling_mean(patient_id, "glucose_mgdl", 12)
        hr   = self.F.get_rolling_mean(patient_id, "heart_rate", 3)
        spo2 = self.F.get_rolling_mean(patient_id, "spo2", 3)
        sbp_trend = self.F.get_trend(patient_id, "sbp", 6)
        glc_trend = self.F.get_trend(patient_id, "glucose_mgdl", 12)
        return np.array([sbp, glc, hr, spo2, sbp_trend, glc_trend],
                         dtype=np.float32)

    # ------------------------------------------------------------------
    # Step 2: Retrieve context from KG + policy (line 2)
    # ------------------------------------------------------------------
    def retrieve_context(self, patient: dict) -> dict:
        conditions = patient.get("conditions", [])
        meds = [m["drug"] for m in patient.get("prescriptions", [])]
        context = {
            "sodium_cap": self.P.get_sodium_cap(
                "hypertension" if "hypertension" in conditions else "healthy"),
            "interactions": [
                self.K.get_drug_interactions(m) for m in meds],
            "contraindications": [
                self.K.get_condition_contraindications(c) for c in conditions],
        }
        return context

    # ------------------------------------------------------------------
    # Step 3: Detect events (line 3)
    # ------------------------------------------------------------------
    def detect_events(self, x: np.ndarray, vitals: dict,
                       adherence: dict) -> list[dict]:
        events = []
        sbp = vitals.get("sbp", 120)
        glc = vitals.get("glucose_mgdl", 100)
        adh_med = adherence.get("medication", 0.7)

        if self.P.should_escalate(vitals):
            events.append({"type": "acute_vital_exceedance", "vitals": vitals})
        if self.P.should_watch(vitals):
            events.append({"type": "watch_zone", "vitals": vitals})
        if adh_med < 0.5:
            events.append({"type": "adherence_lapse",
                            "adherence_score": adh_med})
        if sbp > 150 and x[4] > 0.5:  # rising BP trend
            events.append({"type": "bp_rising_trend", "sbp": sbp,
                            "trend": float(x[4])})
        if glc < 80:
            events.append({"type": "hypoglycemia_risk", "glucose": glc})
        return events

    # ------------------------------------------------------------------
    # Step 4: Formulate goals (line 4)
    # ------------------------------------------------------------------
    def formulate_goals(self, events: list[dict], context: dict) -> list[str]:
        goals = []
        for ev in events:
            t = ev.get("type", "")
            if t == "acute_vital_exceedance":
                goals.append("emergency_escalation")
            elif t == "bp_rising_trend":
                goals.append("normalise_blood_pressure")
            elif t == "adherence_lapse":
                goals.append("improve_medication_adherence")
            elif t == "hypoglycemia_risk":
                goals.append("raise_blood_glucose_safely")
            elif t == "watch_zone":
                goals.append("monitor_and_prepare_escalation")
        if not goals:
            goals.append("maintain_physiological_stability")
        return list(set(goals))  # deduplicate

    # ------------------------------------------------------------------
    # Step 5: Plan tasks (line 5)
    # ------------------------------------------------------------------
    def plan_tasks(self, goals: list[str], context: dict) -> list[dict]:
        tasks = []
        for goal in goals:
            if goal == "emergency_escalation":
                tasks.append({"type": "escalate", "priority": "immediate"})
            elif goal == "normalise_blood_pressure":
                tasks.append({"type": "dietary_modification",
                               "guidance": "low_sodium_meal", "priority": "high"})
                tasks.append({"type": "lifestyle_prompt",
                               "guidance": "reduce_activity_rest", "priority": "high"})
            elif goal == "improve_medication_adherence":
                tasks.append({"type": "medication_reminder", "priority": "routine"})
            elif goal == "raise_blood_glucose_safely":
                tasks.append({"type": "dietary_modification",
                               "guidance": "fast_carb_15g", "priority": "urgent"})
            elif goal == "maintain_physiological_stability":
                tasks.append({"type": "lifestyle_prompt",
                               "guidance": "daily_walk_reminder", "priority": "low"})
        return tasks

    # ------------------------------------------------------------------
    # Main orchestration loop (Listing 1 complete)
    # ------------------------------------------------------------------
    def step(self, patient_id: int, raw_vitals: dict, patient_state: dict,
              observation: Optional[np.ndarray] = None) -> dict:
        """
        One full orchestration cycle:
        Pull → Context → Events → Goals → Tasks → Filter → Resolve → Route → Persist

        Parameters
        ----------
        patient_id   : int
        raw_vitals   : dict — current vital readings
        patient_state: dict — demographics, conditions, prescriptions, adherence
        observation  : np.ndarray | None — RL state vector (25-dim) if using policy

        Returns
        -------
        dict — orchestration result with all routed task outputs
        """
        # --- Listing 1: Lines 1-2 ---
        x = self.pull_latest(patient_id, raw_vitals)
        context = self.retrieve_context(patient_state)

        # --- Line 3: Detect events ---
        adherence = {
            "medication": raw_vitals.get("adherence_med", 0.7),
            "dietary":    raw_vitals.get("adherence_diet", 0.5),
        }
        events = self.detect_events(x, raw_vitals, adherence)

        # --- Line 4: Formulate goals ---
        goals = self.formulate_goals(events, context)

        # --- Line 5: Plan tasks ---
        tasks = self.plan_tasks(goals, context)

        # --- RL policy action (Algorithm 2, line 2) ---
        if self.rl_policy is not None and observation is not None:
            rl_action, _ = self.rl_policy.predict(observation, deterministic=True)
            rl_task = {"type": self.ACTION_NAMES.get(int(rl_action), "no_action"),
                        "source": "rl_policy"}
            if rl_task["type"] != "no_action":
                tasks.append(rl_task)

        # --- Constraint Filter (Algorithm 2, line 3) ---
        filtered_tasks = []
        for task in tasks:
            action_int = list(self.ACTION_NAMES.values()).index(
                task.get("type", "no_action")) if task.get(
                "type") in self.ACTION_NAMES.values() else 0
            filtered_action, reason = self.constraint_filter.filter(
                action_int, raw_vitals,
                adherence["medication"],
                {"hypertension": "hypertension" in patient_state.get("conditions", []),
                 "hour_of_day": (raw_vitals.get("t_minutes", 720) // 60) % 24})
            if reason != "ok":
                task["constraint_applied"] = reason
                task["type"] = self.ACTION_NAMES.get(filtered_action, "no_action")
            filtered_tasks.append(task)

        # --- Conflict Resolver (Algorithm 2, line 4) ---
        resolved_tasks = self.conflict_resolver.resolve(
            filtered_tasks, raw_vitals)

        # --- Route tasks to agents (Algorithm 2, lines 5-8) ---
        results = []
        for task in resolved_tasks:
            task.update({"patient_id": patient_id, **patient_state,
                          "vitals": raw_vitals, **adherence})
            for agent in self.agents.values():
                agent.update_beliefs(task)
            result = self.task_router.route(task, self.agents)
            results.append(result)

        # --- Line 9: Update RL policy feedback ---
        # (Done externally via training loop)

        # --- Persist (Listing 1, line 11) ---
        self.audit_log.append(
            patient_id=str(patient_id),
            agent_id="orchestrator",
            action_type="orchestration_cycle",
            action_detail={
                "goals": goals, "n_tasks": len(resolved_tasks),
                "events": [e["type"] for e in events],
            },
        )

        return {
            "patient_id": patient_id,
            "goals": goals,
            "events": events,
            "tasks_resolved": len(resolved_tasks),
            "results": results,
            "context_summary": {
                "sodium_cap": context["sodium_cap"],
            },
        }

    def log_clinician_override(self, state: dict, proposed_action: int,
                                override_action: int, clinician_id: str,
                                rationale: str):
        """
        Tier 2 HiTL: Log clinician override as 5-tuple for offline policy update.
        Tuple: <s_t, a_proposed, a_override, clinician_id, rationale>
        """
        override_tuple = {
            "state": state,
            "action_proposed": proposed_action,
            "action_override": override_action,
            "clinician_id": clinician_id,
            "rationale": rationale,
        }
        self._override_log.append(override_tuple)
        self.audit_log.append(
            patient_id=str(state.get("patient_id", "unknown")),
            agent_id="clinician_override",
            action_type="tier2_override",
            action_detail=override_tuple,
        )
        return override_tuple
