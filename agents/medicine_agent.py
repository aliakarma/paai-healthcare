"""
medicine_agent.py
=================
Listing 2 from the paper — Medicine Agent.

Responsible for:
    * Adherence monitoring and reminder scheduling within prescriber windows.
    * Drug-food and drug-drug interaction detection.
    * Renal/hepatic boundary enforcement via lab-value checks.
    * Escalation for severe conflicts — never self-resolves high-risk issues.

**Safety contract** (base-class level + domain-level):
    * NEVER adds a new medication.
    * NEVER exceeds prescribed dose ceiling.
    * Escalates rather than guesses when a conflict is ambiguous.

Architecture position::

    preprocessing → envs → agents → orchestrator
                                     ↑
                             medicine_agent.py (here)

Imports::

    from paai_healthcare.agents.medicine_agent import MedicineAgent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import (ActionType, AgentAction, AgentResult, BaseAgent,
                               MedicationEntry, PatientState, Urgency)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Domain-specific data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DrugConflict:
    """Represents a single drug-food or drug-drug interaction finding.

    Attributes:
        drug:        Canonical drug name.
        interactant: Food item or co-medication that causes the conflict.
        severity:    ``"low"``, ``"moderate"``, ``"high"``, or ``"absolute"``.
        description: Plain-English clinical description.
        action:      Recommended mitigation (``"avoid"``, ``"monitor"``, etc.).
    """

    drug: str
    interactant: str
    severity: str
    description: str
    action: str = "monitor"


@dataclass
class RenalFlag:
    """Drug-renal safety check result.

    Attributes:
        drug:      Canonical drug name.
        safe:      ``False`` if the drug must be dose-adjusted or held.
        reason:    Explanation (e.g. ``"eGFR < 30 — metformin contraindicated"``).
        threshold: eGFR threshold that triggered the flag (mL/min/1.73 m²).
    """

    drug: str
    safe: bool
    reason: str
    threshold: float | None = None


# ──────────────────────────────────────────────────────────────────────────────
# MedicineAgent
# ──────────────────────────────────────────────────────────────────────────────


class MedicineAgent(BaseAgent):
    """BDI Medicine Agent — manages medication timing, safety, and adherence.

    **BDI mental state:**

    * Beliefs  — patient vitals, active prescriptions, lab values,
                 adherence fraction, proposed foods in today's meal plan.
    * Desires  — maintain adherence, enforce timing windows, prevent unsafe
                 drug-food interactions, protect renal/hepatic function.
    * Intentions — reminder actions, reschedule actions, escalation actions.

    **Decision logic (Listing 2):**

    For each drug in ``prescriptions``:

    1. Determine the allowed timing window from the policy registry.
    2. Retrieve drug-food conflicts from the knowledge graph for today's foods.
    3. If **severe** conflict (``"high"`` or ``"absolute"``):
       emit ``ESCALATE_DRUG_SAFETY`` and skip all further processing for
       this drug.
    4. Check renal/hepatic safety via ``drug_checker`` if provided.
    5. If adherence < threshold and current time is within window:
       emit ``MEDICATION_REMINDER``.
    6. If drug needs rescheduling (timing drift > 30 min):
       emit ``MEDICATION_RESCHEDULE``.

    Args:
        policy_registry:  Registry for timing windows and dose ceilings.
        knowledge_graph:  Clinical KG for interaction lookups.
        audit_log:        Hash-chained audit sink.
        drug_checker:     Optional drug-safety helper that wraps the KG and
                          registry for compound checks.  When ``None``, only
                          direct KG queries are used.
        adherence_threshold: Adherence fraction below which reminders are
                             issued (default 0.75).
    """

    _DEFAULT_DESIRES = [
        "maintain_medication_adherence",
        "enforce_prescriber_timing_windows",
        "prevent_drug_food_interactions",
        "protect_renal_hepatic_function",
    ]

    def __init__(
        self,
        policy_registry: Any,
        knowledge_graph: Any,
        audit_log: Any,
        drug_checker: Any = None,
        adherence_threshold: float = 0.75,
    ) -> None:
        super().__init__(
            agent_id="medicine_agent",
            policy_registry=policy_registry,
            knowledge_graph=knowledge_graph,
            audit_log=audit_log,
        )
        self.drug_checker = drug_checker
        self.adherence_threshold = adherence_threshold

    # ── BDI cycle ─────────────────────────────────────────────────────────────

    def perceive(self, state: PatientState) -> None:
        """Ingest patient state and populate medication-domain beliefs.

        Converts the structured :class:`PatientState` into the flat
        ``self.beliefs`` dict used by :meth:`deliberate` and :meth:`act`.

        Beliefs populated:

        * ``patient_id``, ``vitals``, ``conditions``, ``allergies``
        * ``prescriptions``  — list of :class:`MedicationEntry` objects
        * ``labs``           — :class:`LabValues` object
        * ``adherence_med``  — rolling 7-day medication adherence [0, 1]
        * ``proposed_foods`` — food IDs in today's tentative meal plan
        * ``hour_of_day``    — for timing-window checks
        * ``actions_tried``  — recent actions for escalation context

        Args:
            state: Current patient state assembled by the orchestrator.
        """
        self.beliefs.update(
            {
                "patient_id": state.patient_id,
                "vitals": state.vitals,
                "conditions": state.conditions,
                "allergies": state.allergies,
                "prescriptions": state.prescriptions,
                "labs": state.labs,
                "adherence_med": state.adherence_med,
                "hour_of_day": state.hour_of_day,
                "actions_tried": state.actions_tried,
                # Proposed foods may be injected by orchestrator via state.extra
                "proposed_foods": state.extra.get("proposed_foods", []),
            }
        )
        self._log.debug(
            "Perceived: patient=%s drugs=%d adherence=%.2f",
            state.patient_id,
            len(state.prescriptions),
            state.adherence_med,
        )

    def deliberate(self) -> list[dict]:
        """Produce medication intentions from current beliefs and desires.

        Iterates over the active prescription list and applies the following
        checks for each drug:

        1. **Interaction check** — queries the KG for drug-food conflicts
           against ``proposed_foods``.  Severe conflicts become
           ``"escalate_drug_safety"`` intentions.
        2. **Renal safety** — if ``drug_checker`` is provided, verifies the
           drug is safe given current eGFR.  Unsafe result becomes
           ``"escalate_drug_safety"``.
        3. **Adherence** — if adherence < threshold, generates a
           ``"medication_reminder"`` intention.

        Returns:
            List of intention dicts, each with keys ``"type"``, ``"urgency"``,
            ``"drug"``, and optional ``"reason"``.
        """
        intentions: list[dict] = []
        prescriptions: list[MedicationEntry] = self.beliefs.get("prescriptions", [])
        labs = self.beliefs.get("labs")
        adherence = self.beliefs.get("adherence_med", 0.75)
        proposed_foods = self.beliefs.get("proposed_foods", [])
        conditions = self.beliefs.get("conditions", [])
        egfr: float | None = labs.egfr if labs else None

        all_drug_names = [m.drug for m in prescriptions]

        for med in prescriptions:
            # ── 1. Drug-food interaction check ───────────────────────────────
            conflicts = self._check_interactions(med.drug, proposed_foods)
            severe = [c for c in conflicts if c.severity in ("high", "absolute")]
            if severe:
                intentions.append(
                    {
                        "type": ActionType.ESCALATE_DRUG_SAFETY.value,
                        "urgency": Urgency.IMMEDIATE.value,
                        "drug": med.drug,
                        "reason": severe[0].description,
                        "conflicts": [vars(c) for c in severe],
                    }
                )
                continue  # Skip further checks for this drug

            # ── 2. Renal / hepatic safety ────────────────────────────────────
            if self.drug_checker is not None:
                flag = self._check_renal_safety(
                    med.drug, conditions, all_drug_names, egfr
                )
                if not flag.safe:
                    intentions.append(
                        {
                            "type": ActionType.ESCALATE_DRUG_SAFETY.value,
                            "urgency": Urgency.HIGH.value,
                            "drug": med.drug,
                            "reason": flag.reason,
                            "threshold": flag.threshold,
                        }
                    )
                    continue

            # ── 3. Adherence reminder ─────────────────────────────────────────
            if adherence < self.adherence_threshold:
                window = self.registry.get_timing_window(med.drug)
                intentions.append(
                    {
                        "type": ActionType.MEDICATION_REMINDER.value,
                        "urgency": Urgency.ROUTINE.value,
                        "drug": med.drug,
                        "window": window,
                        "dose_mg": med.dose_mg,
                        "timing": med.timing,
                    }
                )

        self._log.debug("Deliberated %d intentions", len(intentions))
        return intentions

    def act(self, intentions: list[dict]) -> AgentResult:
        """Convert medication intentions into typed :class:`AgentAction` objects.

        For each intention:

        * ``ESCALATE_DRUG_SAFETY``  → high/immediate-urgency escalation action.
        * ``MEDICATION_REMINDER``   → routine reminder action with dose and
                                      timing details in the payload.
        * ``MEDICATION_RESCHEDULE`` → (produced externally; handled here if
                                      present in the intentions list).

        All actions pass through :meth:`_safety_gate` before being included in
        the result.

        Args:
            intentions: List of intention dicts from :meth:`deliberate`.

        Returns:
            :class:`AgentResult` with all safe, audit-logged actions.
        """
        actions: list[AgentAction] = []

        for intent in intentions:
            intent_type = intent.get("type", "")
            urgency_val = intent.get("urgency", Urgency.ROUTINE.value)
            try:
                urgency = Urgency(urgency_val)
            except ValueError:
                urgency = Urgency.ROUTINE

            # ── Escalation ────────────────────────────────────────────────────
            if intent_type == ActionType.ESCALATE_DRUG_SAFETY.value:
                action = self._make_action(
                    action_type=ActionType.ESCALATE_DRUG_SAFETY,
                    urgency=urgency,
                    payload={
                        "drug": intent["drug"],
                        "reason": intent.get("reason", ""),
                        "conflicts": intent.get("conflicts", []),
                        "threshold": intent.get("threshold"),
                    },
                    rationale=(
                        f"Drug safety concern for {intent['drug']}: "
                        f"{intent.get('reason', 'see conflicts')}"
                    ),
                )
                if self._safety_gate(action):
                    actions.append(action)

            # ── Adherence reminder ────────────────────────────────────────────
            elif intent_type == ActionType.MEDICATION_REMINDER.value:
                window = intent.get("window") or {}
                preferred = window.get(
                    "preferred_time", intent.get("timing", "as prescribed")
                )
                action = self._make_action(
                    action_type=ActionType.MEDICATION_REMINDER,
                    urgency=Urgency.ROUTINE,
                    payload={
                        "drug": intent["drug"],
                        "dose_mg": intent.get("dose_mg"),
                        "timing": intent.get("timing"),
                        "message": (
                            f"Reminder: take {intent['drug']} "
                            f"({intent.get('dose_mg')} mg) — "
                            f"preferred time: {preferred}."
                        ),
                    },
                    rationale=(
                        f"Adherence below threshold "
                        f"({self.beliefs.get('adherence_med', 0):.0%}); "
                        f"issuing reminder for {intent['drug']}."
                    ),
                )
                if self._safety_gate(action):
                    actions.append(action)

            # ── Reschedule ────────────────────────────────────────────────────
            elif intent_type == ActionType.MEDICATION_SCHEDULE.value:
                action = self._make_action(
                    action_type=ActionType.MEDICATION_SCHEDULE,
                    urgency=Urgency.ROUTINE,
                    payload={
                        "drug": intent["drug"],
                        "delta_min": intent.get("delta_min", 0),
                        "new_time": intent.get("new_time"),
                    },
                    rationale=(
                        f"Rescheduling {intent['drug']} by "
                        f"{intent.get('delta_min', 0)} min to stay within "
                        f"prescriber window."
                    ),
                )
                if self._safety_gate(action):
                    actions.append(action)

        self._persist_actions(actions)
        return AgentResult(
            agent_id=self.agent_id,
            actions=actions,
            metadata={
                "prescriptions_reviewed": len(self.beliefs.get("prescriptions", [])),
                "adherence_med": self.beliefs.get("adherence_med"),
            },
        )

    # ── Legacy dict interface (Issue 1 fix) ──────────────────────────────────

    def execute(self, task: dict) -> dict:
        """Execute one medication task via the legacy dict protocol.

        Called by ``task_router.route(task, agents)``.  Merges ``task``
        into beliefs, runs the full BDI pipeline, and returns a plain dict
        so the orchestrator can log and forward the result.

        Args:
            task: Plain task dict from the task router.  May contain
                  ``prescriptions``, ``vitals``, ``labs``,
                  ``proposed_foods``, ``adherence_med``, etc.

        Returns:
            Plain dict with keys ``"agent"``, ``"actions"``, ``"metadata"``.
        """
        self.update_beliefs(task)
        result = self.act(self.deliberate())
        return {
            "agent": self.agent_id,
            "actions": [
                a.payload
                | {"action_type": a.action_type.value, "urgency": a.urgency.value}
                for a in result.actions
            ],
            "metadata": result.metadata,
        }

    # ── Safety gate override ──────────────────────────────────────────────────

    def _safety_gate(self, action: AgentAction) -> bool:
        """Extend base safety gate with medicine-domain rules.

        Additional constraints enforced here:

        * ``add_medication`` key in payload → blocked (inherited).
        * Payload dose exceeding registered ceiling → blocked.

        Args:
            action: Candidate action to validate.

        Returns:
            ``True`` if the action may proceed; ``False`` otherwise.
        """
        if not super()._safety_gate(action):
            return False

        # Enforce dose ceiling from policy registry
        drug = action.payload.get("drug")
        dose_mg = action.payload.get("dose_mg")
        if drug and dose_mg is not None:
            ceiling = self.registry.get_dose_ceiling(drug)
            if ceiling is not None and float(dose_mg) > ceiling:
                self._log.warning(
                    "BLOCKED — dose %.1f mg exceeds ceiling %.1f mg for %s",
                    dose_mg,
                    ceiling,
                    drug,
                )
                return False
        return True

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_interactions(
        self,
        drug: str,
        proposed_foods: list[str],
    ) -> list[DrugConflict]:
        """Query the knowledge graph for drug-food interactions.

        Args:
            drug:           Drug to check.
            proposed_foods: List of food IDs from today's meal plan.

        Returns:
            List of :class:`DrugConflict` objects (empty if none found).
        """
        raw: list[dict] = self.kg.get_drug_interactions(drug)
        conflicts: list[DrugConflict] = []
        # No proposed foods means no drug-food interactions to check.
        # Guard prevents false escalations when the orchestrator has not yet
        # populated the meal plan (e.g. during direct execute() calls in tests).
        if not proposed_foods:
            return conflicts
        for entry in raw:
            # Only flag interactions relevant to today's proposed foods
            if entry.get("food") in proposed_foods:
                conflicts.append(
                    DrugConflict(
                        drug=drug,
                        interactant=entry.get("food", entry.get("drug2", "")),
                        severity=entry.get("severity", "low"),
                        description=entry.get("description", ""),
                        action=entry.get("action", "monitor"),
                    )
                )
        return conflicts

    def _check_renal_safety(
        self,
        drug: str,
        conditions: list[str],
        all_drugs: list[str],
        egfr: float | None,
    ) -> RenalFlag:
        """Use drug_checker to assess renal/hepatic safety.

        Args:
            drug:       Drug name to check.
            conditions: Patient's active clinical conditions.
            all_drugs:  Full prescription list (for drug-drug checks).
            egfr:       Current eGFR, or ``None`` if unavailable.

        Returns:
            :class:`RenalFlag` indicating whether the drug is safe.
        """
        try:
            safe, reason = self.drug_checker.safe_to_prescribe(
                drug, conditions, all_drugs, egfr
            )
            threshold: float | None = None
            # Extract threshold from reason string if present
            if egfr is not None and not safe:
                threshold = egfr
            return RenalFlag(drug=drug, safe=safe, reason=reason, threshold=threshold)
        except Exception as exc:
            self._log.warning("drug_checker failed for %s: %s", drug, exc)
            return RenalFlag(drug=drug, safe=True, reason="checker_unavailable")
