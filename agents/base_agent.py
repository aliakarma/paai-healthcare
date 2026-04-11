"""
base_agent.py
=============
Abstract BDI (Belief-Desire-Intention) base class for all PAAI domain agents.

Architecture position: agents → orchestrator (downstream)
All four domain agents (Medicine, Nutrition, Lifestyle, Emergency) inherit
this class and implement the BDI cognitive cycle:

    perceive()   — update Beliefs from incoming patient state
    deliberate() — derive Intentions from Beliefs × Desires
    act()        — execute one or more Intentions and return AgentResult

The base class enforces the safety contract: no agent may autonomously add
a new medication or trigger an irreversible clinical action.

Imports used downstream::

    from paai_healthcare.agents.base_agent import (
        BaseAgent, PatientState, AgentAction, AgentResult, ActionType
    )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared enumerations
# ──────────────────────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    """Canonical action types produced by domain agents.

    String values are intentionally aligned with ``TaskRouter.TASK_AGENT_MAP``
    so that ``task_router.route(task, agents)`` dispatches to the correct
    agent without a silent fall-through to the emergency agent.

    **TaskRouter-routed values** — string must exactly match the key in
    ``orchestrator/task_router.py::TaskRouter.TASK_AGENT_MAP``:

    +-------------------------------+-----------------------------+
    | Enum member                   | TaskRouter key (= value)    |
    +===============================+=============================+
    | MEDICATION_REMINDER           | medication_reminder         |
    | MEDICATION_SCHEDULE           | medication_schedule         |
    | ESCALATE_DRUG_SAFETY          | escalate_drug_safety        |
    | MEAL_PLAN                     | meal_plan                   |
    | DIETARY_MODIFICATION          | dietary_modification        |
    | SODIUM_ADVISORY               | sodium_advisory             |
    | LIFESTYLE_PROMPT              | lifestyle_prompt            |
    | SLEEP_ADJUSTMENT              | sleep_adjustment            |
    | WALK_PROMPT                   | walk_prompt                 |
    | ESCALATE                      | escalate                    |
    | ESCALATE_TO_CLINICIAN         | escalate_to_clinician       |
    | REPEAT_MEASUREMENT            | repeat_measurement          |
    +-------------------------------+-----------------------------+

    **Patient-app-only values** (not routed through TaskRouter):
        MEAL_SWAP, NAP_RECOMMENDATION, CAFFEINE_HYGIENE,
        DE_ESCALATE, SELF_CARE_GUIDANCE, NO_ACTION
    """

    # ── Medicine domain — TaskRouter-routed ──────────────────────────────────
    MEDICATION_REMINDER = "medication_reminder"
    MEDICATION_SCHEDULE = "medication_schedule"  # was "medication_reschedule" (Issue 2)
    ESCALATE_DRUG_SAFETY = "escalate_drug_safety"

    # ── Nutrition domain — TaskRouter-routed ─────────────────────────────────
    MEAL_PLAN = "meal_plan"
    DIETARY_MODIFICATION = "dietary_modification"  # was "dietary_restriction" (Issue 2)
    SODIUM_ADVISORY = "sodium_advisory"  # added (Issue 2)

    # ── Lifestyle domain — TaskRouter-routed ─────────────────────────────────
    LIFESTYLE_PROMPT = "lifestyle_prompt"  # added (Issue 2)
    SLEEP_ADJUSTMENT = "sleep_adjustment"  # was "sleep_advance" (Issue 2)
    WALK_PROMPT = "walk_prompt"

    # ── Emergency domain — TaskRouter-routed ─────────────────────────────────
    ESCALATE = "escalate"  # added (Issue 2)
    ESCALATE_TO_CLINICIAN = (
        "escalate_to_clinician"  # was "escalate_clinician" (Issue 2)
    )
    REPEAT_MEASUREMENT = "repeat_measurement"

    # ── Patient-app-only nudges (bypass TaskRouter) ───────────────────────────
    MEAL_SWAP = "meal_swap"
    NAP_RECOMMENDATION = "nap_recommendation"
    CAFFEINE_HYGIENE = "caffeine_hygiene"
    SELF_CARE_GUIDANCE = "self_care_guidance"
    DE_ESCALATE = "de_escalate"
    NO_ACTION = "no_action"


class Urgency(str, Enum):
    """Priority level attached to each agent action."""

    IMMEDIATE = "immediate"  # Must route to emergency agent
    HIGH = "high"  # Alert within 60 s
    ROUTINE = "routine"  # Next scheduled digest
    GENTLE = "gentle"  # Soft nudge, no alert


# ──────────────────────────────────────────────────────────────────────────────
# Structured data contracts
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class VitalSigns:
    """Snapshot of a patient's physiological measurements.

    All fields use SI or clinical convention units as described.
    ``None`` means the sensor reading is unavailable at this timestep.
    """

    sbp: float | None = None  # Systolic BP (mmHg)
    dbp: float | None = None  # Diastolic BP (mmHg)
    glucose_mgdl: float | None = None  # Blood glucose (mg/dL)
    heart_rate: float | None = None  # Heart rate (bpm)
    spo2: float | None = None  # Peripheral O₂ saturation (%)
    temperature_c: float | None = None  # Body temperature (°C)


@dataclass
class LabValues:
    """Relevant laboratory results used for drug-safety checks."""

    egfr: float | None = None  # eGFR (mL/min/1.73 m²) — renal function
    ast: float | None = None  # AST (U/L) — hepatic
    alt: float | None = None  # ALT (U/L) — hepatic
    potassium_meq: float | None = None  # Serum K⁺ (mEq/L)
    sodium_meq: float | None = None  # Serum Na⁺ (mEq/L)
    hba1c: float | None = None  # HbA1c (%)
    ldl_mgdl: float | None = None  # LDL-C (mg/dL)


@dataclass
class MedicationEntry:
    """Single medication in the patient's active prescription list."""

    drug: str  # DrugBank canonical name
    dose_mg: float  # Current prescribed dose
    frequency: str  # e.g. "once_daily", "twice_daily"
    timing: str  # e.g. "morning", "with_meals", "bedtime"
    route: str = "oral"  # Administration route
    notes: str = ""  # Renal/hepatic precaution free text


@dataclass
class PatientState:
    """Unified patient state passed to each agent's ``perceive()`` method.

    This dataclass consolidates all signals the orchestrator has assembled:
    vitals from the feature store, demographics from EHR, current
    prescriptions, adherence metrics, and applicable policy flags.  Agents
    must not mutate this object — all updates happen through ``self.beliefs``.

    Attributes:
        patient_id:       Pseudonymised patient identifier (SHA-256 prefix).
        vitals:           Current vital sign snapshot.
        labs:             Most recent lab values (may be days old).
        prescriptions:    Active medication list from EHR.
        conditions:       ICD-10-coded active diagnoses (lowercase slug).
        allergies:        Allergy map, e.g. ``{"penicillin": True}``.
        adherence_med:    7-day rolling medication adherence fraction [0, 1].
        adherence_diet:   7-day rolling dietary adherence fraction [0, 1].
        steps_today:      Pedometer step count for today.
        sleep_actual_hours: Last night's total sleep time.
        sleep_target_hours: Personalised sleep target from chronotype model.
        chronotype:       Patient chronotype: ``"morning"``, ``"intermediate"``,
                          or ``"evening"``.
        hour_of_day:      Current hour (0-23) in patient local time.
        caffeine_intake_mg: Estimated daily caffeine consumed so far.
        trends:           Rolling statistics dict from feature store.
        actions_tried:    List of recent agent actions (for escalation context).
        extra:            Arbitrary extension dict for future fields.
    """

    patient_id: str

    # Physiological signals
    vitals: VitalSigns = field(default_factory=VitalSigns)
    labs: LabValues = field(default_factory=LabValues)

    # Clinical record
    prescriptions: list[MedicationEntry] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    allergies: dict[str, bool] = field(default_factory=dict)

    # Adherence
    adherence_med: float = 0.75
    adherence_diet: float = 0.55

    # Lifestyle signals
    steps_today: int = 0
    sleep_actual_hours: float = 7.0
    sleep_target_hours: float = 7.5
    chronotype: str = "intermediate"
    hour_of_day: int = 12
    caffeine_intake_mg: float = 0.0

    # Context from feature store / orchestrator
    trends: dict[str, float] = field(default_factory=dict)
    actions_tried: list[dict] = field(default_factory=list)

    # Open extension slot — orchestrator may attach extra context here
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """A single concrete action recommended by a domain agent.

    Attributes:
        action_type:  One of the :class:`ActionType` enum values.
        urgency:      How quickly this action must be delivered.
        payload:      Action-specific structured data (message text, drug name,
                      meal plan dict, etc.).
        agent_id:     Identifier of the agent that produced this action.
        patient_id:   Target patient pseudonym.
        rationale:    Human-readable explanation for HiTL audit trail.
    """

    action_type: ActionType
    urgency: Urgency
    payload: dict[str, Any]
    agent_id: str
    patient_id: str
    rationale: str = ""


@dataclass
class AgentResult:
    """Aggregated output returned by ``act()``.

    The orchestrator collects :class:`AgentResult` objects from each agent it
    dispatches and merges them before routing to patient/clinician outputs.

    Attributes:
        agent_id:  Producing agent identifier.
        actions:   Ordered list of recommended actions.
        metadata:  Diagnostic metadata (plan summaries, targets, etc.).
    """

    agent_id: str
    actions: list[AgentAction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Dependency protocols (structural typing — avoids circular imports)
# ──────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class PolicyRegistryProtocol(Protocol):
    """Structural type expected by agents for policy lookups.

    Every method here must be implemented by the concrete
    ``knowledge.policy_registry.PolicyRegistry`` class.  Agents depend only
    on this protocol, not the concrete class, to prevent circular imports.
    """

    def should_escalate(self, vitals: dict) -> bool: ...
    def should_watch(self, vitals: dict) -> bool: ...
    def get_sodium_cap(self, condition: str) -> float: ...
    def get_timing_window(self, drug: str) -> dict | None: ...
    def get_dose_ceiling(self, drug: str) -> float | None: ...

    def get_caffeine_cutoff(self, condition: str) -> int:
        """Return the hour (0-23) after which caffeine should be avoided.

        Added to resolve Issue 3: ``LifestyleAgent._caffeine_cutoff_hour()``
        previously bypassed this protocol by calling
        ``getattr(self.registry, "rules", {})`` directly, which is a concrete
        attribute of ``PolicyRegistry`` not visible through the protocol.
        Any protocol-conformant mock silently returned hardcoded 14.

        Concrete implementation reads from
        ``rules["caffeine_restriction"][condition]["after_hour"]``.

        Args:
            condition: Patient condition key, e.g. ``"hypertension"`` or
                       ``"healthy"``.

        Returns:
            Integer hour in [0, 23].  14 is the evidence-based default.
        """
        ...


@runtime_checkable
class KnowledgeGraphProtocol(Protocol):
    """Structural type expected by agents for clinical KG queries."""

    def get_drug_interactions(self, drug: str) -> list[dict]: ...
    def check_plan_conflicts(self, meds: list, foods: list) -> list[dict]: ...
    def get_condition_contraindications(self, condition: str) -> list[dict]: ...


@runtime_checkable
class AuditLogProtocol(Protocol):
    """Structural type for hash-chained audit persistence."""

    def append(
        self,
        patient_id: str,
        agent_id: str,
        action_type: str,
        action_detail: dict,
        outcome: dict,
    ) -> str: ...


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base agent
# ──────────────────────────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """Abstract BDI agent implementing the three-stage cognitive cycle.

    All domain agents (Medicine, Nutrition, Lifestyle, Emergency) inherit this
    class.  The BDI mental-state model is represented as:

    * **Beliefs**    — ``self.beliefs`` dict, updated by :meth:`perceive`.
    * **Desires**    — ``self.desires`` list of clinical goal strings, set by
                       the orchestrator via :meth:`set_desires`.
    * **Intentions** — intermediate list of action-intent dicts produced by
                       :meth:`deliberate` and consumed by :meth:`act`.

    The canonical execution order per timestep is::

        agent.perceive(state)       # ingest patient state → beliefs
        intents = agent.deliberate()# beliefs × desires → intentions
        result  = agent.act(intents)# intentions → AgentResult

    Subclasses MUST implement all three abstract methods.  They MUST NOT:

    * Add new medications autonomously (enforced by :meth:`_safety_gate`).
    * Call RL policy methods (RL lives in the ``rl`` package).
    * Import from ``paai_healthcare.orchestrator`` (would create a cycle).

    Args:
        agent_id:         Unique string identifier, e.g. ``"medicine_agent"``.
        policy_registry:  Implements :class:`PolicyRegistryProtocol`.
        knowledge_graph:  Implements :class:`KnowledgeGraphProtocol`.
        audit_log:        Implements :class:`AuditLogProtocol`.
    """

    # Subclasses declare their static desires here
    _DEFAULT_DESIRES: list[str] = []

    def __init__(
        self,
        agent_id: str,
        policy_registry: PolicyRegistryProtocol,
        knowledge_graph: KnowledgeGraphProtocol,
        audit_log: AuditLogProtocol,
    ) -> None:
        self.agent_id = agent_id
        self.registry = policy_registry
        self.kg = knowledge_graph
        self.audit_log = audit_log

        # ── BDI mental state ─────────────────────────────────────────────────
        self.beliefs: dict[str, Any] = {}
        self.desires: list[str] = list(self.__class__._DEFAULT_DESIRES)
        self.intentions: list[dict] = []

        self._log = logging.getLogger(f"paai_healthcare.agents.{agent_id}")

    # ── Legacy dict interface (Issue 1 fix) ──────────────────────────────────

    def update_beliefs(self, new_state: dict) -> None:
        """Merge a raw task/state dict into ``self.beliefs``.

        This is the entry point called by the orchestrator before routing::

            agent.update_beliefs(task)   # orchestrator.step(), line 234
            result = agent.execute(task) # task_router.route()

        Non-abstract so all subclasses inherit it without override.
        Subclasses may call ``super().update_beliefs(state)`` and then
        extract typed fields into domain-specific belief keys.

        Args:
            new_state: Plain dict from the orchestrator task pipeline.
        """
        self.beliefs.update(new_state)

    @abstractmethod
    def execute(self, task: dict) -> dict:
        """Execute one routed task and return a plain-dict result.

        Called by ``task_router.route(task, agents)`` which expects a plain
        ``dict`` back — **not** an :class:`AgentResult`.

        Implementations must:

        1. Call ``self.update_beliefs(task)`` to merge task context.
        2. Perform domain logic (may delegate to ``deliberate()`` / ``act()``
           or work inline for simplicity).
        3. Return a serialisable ``dict``, at minimum
           ``{"agent": self.agent_id, "actions": [...]}``.

        Args:
            task: Plain dict from the task router.

        Returns:
            Plain ``dict`` result, not an :class:`AgentResult`.
        """

    # ── Structured BDI cycle ──────────────────────────────────────────────────

    @abstractmethod
    def perceive(self, state: PatientState) -> None:
        """Ingest a :class:`PatientState` and update ``self.beliefs``.

        This method converts the structured ``state`` dataclass into the flat
        ``beliefs`` dict used by :meth:`deliberate` and :meth:`act`.  Each
        subclass may populate agent-specific belief entries beyond the common
        ones written here.

        Args:
            state: Full patient state assembled by the orchestrator.
        """

    @abstractmethod
    def deliberate(self) -> list[dict]:
        """Map current Beliefs × Desires to Intentions.

        Produces a list of *intention dicts* — lightweight action-intent
        descriptors (not full :class:`AgentAction` objects).  These are
        passed directly to :meth:`act`.

        Each intention dict must contain at least:
        * ``"type"``: an :class:`ActionType` string value.
        * ``"urgency"``: an :class:`Urgency` string value.

        Returns:
            Ordered list of intention dicts (empty list if no action needed).
        """

    @abstractmethod
    def act(self, intentions: list[dict]) -> AgentResult:
        """Convert Intentions into concrete :class:`AgentAction` objects.

        Implementations must:
        1. Iterate over ``intentions``.
        2. Build a typed :class:`AgentAction` per intention.
        3. Call :meth:`_safety_gate` on every action before appending it.
        4. Persist final actions via :meth:`_persist_actions`.

        Args:
            intentions: List of intention dicts from :meth:`deliberate`.

        Returns:
            :class:`AgentResult` containing all safe, loggable actions.
        """

    # ── Orchestrator integration hooks ────────────────────────────────────────

    def set_desires(self, goals: list[str]) -> None:
        """Override the current desire set (called by orchestrator).

        Args:
            goals: Replacement list of clinical goal strings.
        """
        self.desires = goals
        self._log.debug("Desires updated: %s", goals)

    def run(self, state: PatientState) -> AgentResult:
        """Execute the full BDI cycle in a single call.

        This is the primary entrypoint used by the orchestrator::

            result = agent.run(patient_state)

        Equivalent to::

            agent.perceive(state)
            intents = agent.deliberate()
            result  = agent.act(intents)

        Args:
            state: Current patient state from the orchestrator.

        Returns:
            :class:`AgentResult` ready for downstream routing.
        """
        self.perceive(state)
        self.intentions = self.deliberate()
        return self.act(self.intentions)

    # ── Safety contract ───────────────────────────────────────────────────────

    def _safety_gate(self, action: AgentAction) -> bool:
        """Return ``False`` and log a warning if an action violates policy.

        Hard rules enforced at base-class level:
        * :attr:`ActionType.ESCALATE_DRUG_SAFETY` with ``"add_medication"``
          semantics is blocked — no agent may add a new drug.
        * Any action whose ``payload`` contains the key ``"add_medication"``
          is blocked unconditionally.

        Subclasses may call ``super()._safety_gate(action)`` and then add
        domain-specific checks.

        Args:
            action: Candidate :class:`AgentAction` to validate.

        Returns:
            ``True`` if the action is permissible, ``False`` otherwise.
        """
        if action.payload.get("add_medication"):
            self._log.warning(
                "BLOCKED — agent '%s' attempted to add a new medication "
                "(autonomous prescribing is prohibited).  patient_id=%s",
                self.agent_id,
                action.patient_id,
            )
            return False
        return True

    # ── Audit persistence ─────────────────────────────────────────────────────

    def _persist_actions(self, actions: list[AgentAction]) -> None:
        """Write all safe actions to the hash-chained audit log.

        Args:
            actions: Final list of :class:`AgentAction` objects produced by
                     :meth:`act`.
        """
        for action in actions:
            try:
                self.audit_log.append(
                    patient_id=action.patient_id,
                    agent_id=self.agent_id,
                    action_type=action.action_type.value,
                    action_detail=action.payload,
                    outcome={
                        "urgency": action.urgency.value,
                        "rationale": action.rationale,
                    },
                )
            except Exception as exc:  # never let logging crash the agent
                self._log.error("Audit log write failed: %s", exc)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def _make_action(
        self,
        action_type: ActionType,
        urgency: Urgency,
        payload: dict[str, Any],
        rationale: str = "",
    ) -> AgentAction:
        """Construct an :class:`AgentAction` pre-filled with agent metadata.

        Args:
            action_type: Canonical action type enum value.
            urgency:     Delivery urgency.
            payload:     Action-specific data dict.
            rationale:   Optional explanation for the audit trail.

        Returns:
            Fully populated :class:`AgentAction`.
        """
        return AgentAction(
            action_type=action_type,
            urgency=urgency,
            payload=payload,
            agent_id=self.agent_id,
            patient_id=str(self.beliefs.get("patient_id", "unknown")),
            rationale=rationale,
        )

    def _vitals_dict(self) -> dict[str, float]:
        """Return a plain-dict copy of current vitals beliefs for registry calls."""
        vitals_raw = self.beliefs.get("vitals", {})
        if isinstance(vitals_raw, VitalSigns):
            return {
                "sbp": vitals_raw.sbp or 0.0,
                "dbp": vitals_raw.dbp or 0.0,
                "glucose_mgdl": vitals_raw.glucose_mgdl or 0.0,
                "heart_rate": vitals_raw.heart_rate or 0.0,
                "spo2": vitals_raw.spo2 or 100.0,
            }
        return dict(vitals_raw)  # already a dict (legacy path)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"agent_id={self.agent_id!r}, "
            f"desires={self.desires!r})"
        )


# Backward-compatibility alias expected by verify_merge.py and tests
BDIAgent = BaseAgent
