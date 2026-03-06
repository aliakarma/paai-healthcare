"""
emergency_agent.py
==================
Listing 5 / Algorithm 3 from the paper — Emergency Escalation Agent.

Responsible for:
    * Detecting acute physiological exceedances (e.g. SBP ≥ 180 mmHg,
      glucose ≤ 54 mg/dL, SpO₂ ≤ 90%).
    * Implementing the **watch-and-repeat protocol** — scheduling two
      confirmatory measurements (at +20 min and +40 min) before escalating.
    * Composing rich, consent-verified alert packets so clinicians can act
      without re-collecting data.
    * Providing interim self-care guidance while awaiting repeat measurements.
    * De-escalating cleanly if readings normalise on re-measurement.

**Watch-and-repeat protocol (Algorithm 3):**

::

    if AcuteExceedance(x, Γ) ∨ SevereSymptoms(Σ):
        ScheduleRepeatMeasures(20, 40 min)
        if ConfirmedPersistent(x, 2 repeats):
            pkt = ComposeAlert(x, trends, medsOnBoard, actionsTried)
            SecureSend(Clinician, pkt)        ← consent-checked, encrypted
        else:
            ProvideSelfCare(rest, hydration, posture)

**State management:**
The agent maintains a per-patient repeat-measurement state dict in
``_pending_repeats``.  This dict is cleared on escalation, de-escalation,
or patient discharge.  When running inside a stateless microservice,
persistence should be offloaded to the feature store.

Architecture position::

    preprocessing → envs → agents → orchestrator
                                     ↑
                             emergency_agent.py (here)

Imports::

    from paai_healthcare.agents.emergency_agent import EmergencyAgent
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from agents.base_agent import (
    ActionType,
    AgentAction,
    AgentResult,
    BaseAgent,
    PatientState,
    Urgency,
    # VitalSigns removed — imported but never referenced (Issue 5)
)

logger = logging.getLogger(__name__)

# Repeat-measurement schedule intervals (minutes)
REPEAT_INTERVALS_MINUTES: tuple[int, int] = (20, 40)

# Number of confirmatory readings required before escalation
PERSISTENCE_THRESHOLD: int = 2

# Glucose threshold for fast-acting carbohydrate self-care guidance (mg/dL)
HYPOGLYCAEMIA_SELF_CARE_THRESHOLD: float = 70.0

# SBP threshold for hypertensive self-care guidance (mmHg)
HYPERTENSION_SELF_CARE_THRESHOLD: float = 160.0


# ──────────────────────────────────────────────────────────────────────────────
# Domain-specific data structures
# ──────────────────────────────────────────────────────────────────────────────


class EscalationStatus(Enum):
    """Outcome of a single call to :meth:`EmergencyAgent.act`."""

    NO_ACTION = auto()  # Vitals within safe range
    REPEAT_SCHEDULED = auto()  # First or second repeat measurement queued
    ESCALATED_TO_CLINICIAN = auto()  # Persistent — alert sent
    DE_ESCALATED = auto()  # Resolved on re-measurement


@dataclass
class RepeatMeasurementState:
    """In-memory state for the watch-and-repeat protocol per patient.

    Attributes:
        count:             Number of repeat measurements completed so far.
        last_vitals_dict:  Vitals snapshot that triggered the protocol.
        created_at:        Unix timestamp when the protocol was initiated.
    """

    count: int = 0
    last_vitals_dict: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class AlertPacket:
    """Rich context packet sent to the clinician on confirmed escalation.

    Attributes:
        patient_pseudonym:     SHA-256-prefixed patient identifier.
        timestamp:             Unix timestamp of escalation decision.
        triggering_vitals:     The vital readings that triggered escalation.
        recent_trends:         Rolling statistics from the feature store.
        medications_on_board:  Full prescription list at time of alert.
        adherence_last_7days:  Adherence history for clinical context.
        recent_actions_tried:  Agent actions taken prior to escalation.
        self_care_provided:    Self-care guidance already given to patient.
        consent_verified:      Whether the patient has consented to alerts.
        encrypted:             Whether the packet is AES-256 encrypted.
        repeat_count:          Number of confirmatory readings obtained.
    """

    patient_pseudonym: str
    timestamp: float
    triggering_vitals: dict[str, Any]
    recent_trends: dict[str, Any]
    medications_on_board: list[Any]
    adherence_last_7days: dict[str, Any]
    recent_actions_tried: list[dict]
    self_care_provided: list[str]
    consent_verified: bool = True
    encrypted: bool = True
    repeat_count: int = PERSISTENCE_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# EmergencyAgent
# ──────────────────────────────────────────────────────────────────────────────


class EmergencyAgent(BaseAgent):
    """BDI Emergency Escalation Agent — watch-and-repeat before alert.

    **BDI mental state:**

    * Beliefs  — latest vitals, trends, medications on board, repeat-
                 measurement history, symptoms, consent status.
    * Desires  — maximise detection of true emergencies (high sensitivity),
                 minimise false-positive escalations (high specificity).
    * Intentions — ``initiate_repeat_measurement`` | ``escalate_clinician`` |
                   ``provide_self_care`` | ``de_escalate``.

    **Repeat state lifecycle:**

    ::

        Normal → [acute signal] → REPEAT_1 (+20 min) → REPEAT_2 (+40 min)
              → still_abnormal? → ESCALATED
              → normalised?     → DE_ESCALATED → Normal

    The ``_pending_repeats`` dict maps ``patient_id`` →
    :class:`RepeatMeasurementState`.  It is cleared after any terminal
    transition (escalate or de-escalate).

    Args:
        policy_registry:      Registry for escalation thresholds and watch zones.
        knowledge_graph:      Clinical KG (available for future drug-in-crisis
                              checks; not used in this version).
        audit_log:            Hash-chained audit sink.
        persistence_threshold: Number of confirmatory abnormal readings required
                               before escalation (default 2).
    """

    _DEFAULT_DESIRES = [
        "detect_acute_physiological_emergencies",
        "minimise_false_positive_escalations",
        "provide_actionable_self_care_guidance",
        "compose_rich_clinician_alert_packets",
    ]

    def __init__(
        self,
        policy_registry: Any,
        knowledge_graph: Any,
        audit_log: Any,
        persistence_threshold: int = PERSISTENCE_THRESHOLD,
    ) -> None:
        super().__init__(
            agent_id="emergency_agent",
            policy_registry=policy_registry,
            knowledge_graph=knowledge_graph,
            audit_log=audit_log,
        )
        self._persistence_threshold = persistence_threshold
        # In-memory per-patient repeat state (offload to feature store in prod)
        self._pending_repeats: dict[str, RepeatMeasurementState] = {}

    # ── BDI cycle ─────────────────────────────────────────────────────────────

    def perceive(self, state: PatientState) -> None:
        """Ingest patient state and populate emergency-domain beliefs.

        Beliefs populated:

        * ``patient_id``           — for repeat-state lookup
        * ``vitals``               — VitalSigns + flat vitals_dict for registry
        * ``trends``               — rolling stats for alert packet
        * ``prescriptions``        — medications on board for context
        * ``adherence_history``    — 7-day adherence for alert packet
        * ``actions_tried``        — prior agent actions for escalation context
        * ``severe_symptoms``      — bool from ``state.extra``
        * ``consent_alerts``       — bool from ``state.extra`` (default True)

        Args:
            state: Current patient state from the orchestrator.
        """
        vitals = state.vitals
        vitals_dict = {
            "sbp": vitals.sbp,
            "dbp": vitals.dbp,
            "glucose_mgdl": vitals.glucose_mgdl,
            "heart_rate": vitals.heart_rate,
            "spo2": vitals.spo2,
        }

        self.beliefs.update(
            {
                "patient_id": state.patient_id,
                "vitals": vitals,
                "vitals_dict": {k: v for k, v in vitals_dict.items() if v is not None},
                "trends": state.trends,
                "prescriptions": state.prescriptions,
                "adherence_history": {
                    "adherence_med": state.adherence_med,
                    "adherence_diet": state.adherence_diet,
                },
                "actions_tried": state.actions_tried,
                "severe_symptoms": state.extra.get("severe_symptoms", False),
                "consent_alerts": state.extra.get("consent_alerts", True),
            }
        )
        self._log.debug(
            "Perceived: patient=%s sbp=%s glucose=%s spo2=%s",
            state.patient_id,
            vitals.sbp,
            vitals.glucose_mgdl,
            vitals.spo2,
        )

    def deliberate(self) -> list[dict]:
        """Determine emergency intentions from current beliefs.

        Logic:

        1. If vitals trigger ``should_escalate`` (acute zone) OR severe
           symptoms are present: emit ``initiate_repeat_measurement``.
        2. Else if vitals are in ``should_watch`` (watch zone): emit
           ``monitor_watch_zone`` (routine monitoring).
        3. Otherwise: no action needed.

        Returns:
            List of zero or one intention dicts.
        """
        vitals_dict = self.beliefs.get("vitals_dict", {})
        severe_symptoms = bool(self.beliefs.get("severe_symptoms", False))

        if self.registry.should_escalate(vitals_dict) or severe_symptoms:
            return [
                {
                    "type": "initiate_repeat_measurement",
                    "urgency": Urgency.IMMEDIATE.value,
                }
            ]
        if self.registry.should_watch(vitals_dict):
            return [
                {
                    "type": "monitor_watch_zone",
                    "urgency": Urgency.HIGH.value,
                }
            ]
        return []

    def act(self, intentions: list[dict]) -> AgentResult:
        """Execute emergency protocol and produce escalation actions.

        For each intention:

        * ``initiate_repeat_measurement`` — runs the full watch-and-repeat
          state machine:

          * If repeat count < threshold: schedule next repeat, return
            ``REPEAT_MEASUREMENT`` action with self-care guidance.
          * If repeat count == threshold AND still abnormal: compose
            ``ESCALATE_CLINICIAN`` action.
          * If repeat count == threshold AND normalised: ``DE_ESCALATE``.

        * ``monitor_watch_zone`` — returns a routine monitoring action
          (no escalation, no self-care needed).

        * No intentions → returns ``NO_ACTION`` result.

        Args:
            intentions: List of intention dicts from :meth:`deliberate`.

        Returns:
            :class:`AgentResult` with appropriate escalation or monitoring
            actions, and ``metadata["status"]`` set to an
            :class:`EscalationStatus` value name.
        """
        if not intentions:
            return AgentResult(
                agent_id=self.agent_id,
                actions=[],
                metadata={"status": EscalationStatus.NO_ACTION.name},
            )

        actions: list[AgentAction] = []
        status = EscalationStatus.NO_ACTION
        pid = str(self.beliefs.get("patient_id", "unknown"))
        vitals_dict = self.beliefs.get("vitals_dict", {})

        for intent in intentions:
            intent_type = intent.get("type", "")

            # ── Watch-zone monitoring (not yet acute) ─────────────────────────
            if intent_type == "monitor_watch_zone":
                action = self._make_action(
                    action_type=ActionType.SELF_CARE_GUIDANCE,
                    urgency=Urgency.HIGH,
                    payload={
                        "message": (
                            "Your readings are in a watch zone.  "
                            "Please rest, stay hydrated, and continue "
                            "monitoring every 15 minutes."
                        ),
                        "vitals": vitals_dict,
                    },
                    rationale="Vitals in watch zone — monitoring without escalation.",
                )
                actions.append(action)
                status = EscalationStatus.NO_ACTION
                continue

            # ── Repeat-measurement state machine ──────────────────────────────
            if intent_type == "initiate_repeat_measurement":
                rep_state = self._pending_repeats.get(pid, RepeatMeasurementState())

                # ── Still collecting confirmatory readings ────────────────────
                if rep_state.count < self._persistence_threshold:
                    rep_state.count += 1
                    rep_state.last_vitals_dict = vitals_dict
                    self._pending_repeats[pid] = rep_state

                    next_interval = REPEAT_INTERVALS_MINUTES[
                        min(rep_state.count - 1, len(REPEAT_INTERVALS_MINUTES) - 1)
                    ]
                    self_care = self._self_care_guidance(vitals_dict)
                    action = self._make_action(
                        action_type=ActionType.REPEAT_MEASUREMENT,
                        urgency=Urgency.HIGH,
                        payload={
                            "repeat_number": rep_state.count,
                            "next_repeat_in_minutes": next_interval,
                            "self_care_guidance": self_care,
                            "vitals_at_trigger": vitals_dict,
                            "message": (
                                f"Confirmatory measurement {rep_state.count}/"
                                f"{self._persistence_threshold} scheduled in "
                                f"{next_interval} minutes.  "
                                f"Follow self-care guidance while waiting."
                            ),
                        },
                        rationale=(
                            f"Watch-and-repeat: repeat {rep_state.count} of "
                            f"{self._persistence_threshold}."
                        ),
                    )
                    actions.append(action)
                    status = EscalationStatus.REPEAT_SCHEDULED

                # ── Persistence check after required repeats ──────────────────
                else:
                    still_abnormal = self.registry.should_escalate(
                        vitals_dict
                    ) or self.registry.should_watch(vitals_dict)

                    if still_abnormal:
                        # ── Escalate ──────────────────────────────────────────
                        self_care = self._self_care_guidance(vitals_dict)
                        packet = self._compose_alert_packet(
                            vitals_dict, self_care, rep_state.count
                        )
                        action = self._make_action(
                            action_type=ActionType.ESCALATE_TO_CLINICIAN,  # was ESCALATE_CLINICIAN (Issue 2)
                            urgency=Urgency.IMMEDIATE,
                            payload={
                                "alert_packet": self._packet_to_dict(packet),
                                "self_care": self_care,
                                "message": (
                                    "Persistent abnormal readings confirmed "
                                    f"after {rep_state.count} measurements.  "
                                    "Secure alert sent to clinician."
                                ),
                            },
                            rationale=(
                                f"Persistent exceedance after "
                                f"{rep_state.count} repeat readings — "
                                "escalating to clinician."
                            ),
                        )
                        self._persist_actions([action])
                        actions.append(action)
                        status = EscalationStatus.ESCALATED_TO_CLINICIAN
                        # Clear repeat state on escalation
                        self._pending_repeats.pop(pid, None)

                    else:
                        # ── De-escalate ───────────────────────────────────────
                        action = self._make_action(
                            action_type=ActionType.DE_ESCALATE,
                            urgency=Urgency.ROUTINE,
                            payload={
                                "vitals": vitals_dict,
                                "message": (
                                    "Readings have returned to within normal "
                                    "range on re-measurement.  No escalation "
                                    "required at this time.  Continue monitoring."
                                ),
                            },
                            rationale=(
                                "Vitals normalised on confirmatory measurement — "
                                "de-escalating."
                            ),
                        )
                        actions.append(action)
                        status = EscalationStatus.DE_ESCALATED
                        self._pending_repeats.pop(pid, None)

        # Only persist non-escalation actions here (escalation already persisted above)
        non_escalation = [
            a
            for a in actions
            if a.action_type
            != ActionType.ESCALATE_TO_CLINICIAN  # was ESCALATE_CLINICIAN (Issue 2)
        ]
        self._persist_actions(non_escalation)

        return AgentResult(
            agent_id=self.agent_id,
            actions=actions,
            metadata={
                "status": status.name,
                "pending_repeat_pid": pid if pid in self._pending_repeats else None,
            },
        )

    # ── Legacy dict interface (Issue 1 fix) ──────────────────────────────────

    def update_beliefs(self, new_state: dict) -> None:
        """Extend base update_beliefs to also populate ``vitals_dict``.

        ``deliberate()`` reads from ``self.beliefs['vitals_dict']`` (a flat
        dict of numeric vital readings), but the dict-based ``execute()``
        path only provides ``new_state['vitals']``.  This override ensures
        that ``vitals_dict`` is always available regardless of whether the
        agent was driven by ``perceive(PatientState)`` or
        ``execute(task_dict)``.

        Args:
            new_state: Plain dict from the orchestrator or test harness.
        """
        super().update_beliefs(new_state)
        # Derive vitals_dict if not already present but vitals is
        if "vitals_dict" not in self.beliefs and "vitals" in self.beliefs:
            raw_vitals = self.beliefs["vitals"]
            if isinstance(raw_vitals, dict):
                self.beliefs["vitals_dict"] = {
                    k: v for k, v in raw_vitals.items() if v is not None
                }

    def execute(self, task: dict) -> dict:
        """Execute one emergency task via the legacy dict protocol.

        Called by ``task_router.route(task, agents)``.  Merges ``task``
        into beliefs, runs the full BDI pipeline, and returns a plain dict.

        Args:
            task: Plain task dict.  Expected to contain ``vitals`` (dict),
                  ``patient_id``, and optionally ``trends``,
                  ``prescriptions``, ``severe_symptoms``.

        Returns:
            Plain dict with keys ``"agent"``, ``"actions"``,
            ``"metadata"`` (includes ``"status"`` from
            :class:`EscalationStatus`).
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compose_alert_packet(
        self,
        vitals_dict: dict[str, Any],
        self_care: list[str],
        repeat_count: int,
    ) -> AlertPacket:
        """Build the rich context packet sent to the clinician.

        Includes all information a clinician needs to act without
        requesting additional data (Algorithm 3, line 4).

        Args:
            vitals_dict:   Current vital readings that confirmed persistence.
            self_care:     Self-care guidance already issued to the patient.
            repeat_count:  Number of confirmatory readings completed.

        Returns:
            :class:`AlertPacket` — consent-verified, flagged for encryption.
        """
        consent = bool(self.beliefs.get("consent_alerts", True))
        return AlertPacket(
            patient_pseudonym=str(self.beliefs.get("patient_id", "unknown")),
            timestamp=time.time(),
            triggering_vitals=vitals_dict,
            recent_trends=dict(self.beliefs.get("trends", {})),
            medications_on_board=list(self.beliefs.get("prescriptions", [])),
            adherence_last_7days=dict(self.beliefs.get("adherence_history", {})),
            recent_actions_tried=list(self.beliefs.get("actions_tried", [])),
            self_care_provided=self_care,
            consent_verified=consent,
            encrypted=True,
            repeat_count=repeat_count,
        )

    def _packet_to_dict(self, packet: AlertPacket) -> dict[str, Any]:
        """Serialise an :class:`AlertPacket` to a JSON-safe dict.

        Converts :class:`MedicationEntry` objects in ``medications_on_board``
        to plain dicts to ensure JSON serialisability.

        Args:
            packet: The alert packet to serialise.

        Returns:
            JSON-serialisable dict representation of the packet.
        """

        def _serialise_med(m: Any) -> dict:
            if hasattr(m, "__dataclass_fields__"):
                return {
                    "drug": m.drug,
                    "dose_mg": m.dose_mg,
                    "frequency": m.frequency,
                    "timing": m.timing,
                    "route": m.route,
                }
            return dict(m) if isinstance(m, dict) else str(m)

        return {
            "patient_pseudonym": packet.patient_pseudonym,
            "timestamp": packet.timestamp,
            "triggering_vitals": packet.triggering_vitals,
            "recent_trends": packet.recent_trends,
            "medications_on_board": [
                _serialise_med(m) for m in packet.medications_on_board
            ],
            "adherence_last_7days": packet.adherence_last_7days,
            "recent_actions_tried": packet.recent_actions_tried,
            "self_care_provided": packet.self_care_provided,
            "consent_verified": packet.consent_verified,
            "encrypted": packet.encrypted,
            "repeat_count": packet.repeat_count,
        }

    def _self_care_guidance(self, vitals_dict: dict[str, Any]) -> list[str]:
        """Generate interim self-care instructions based on vital type.

        Guidance is condition-specific:

        * **Hypertension** (SBP ≥ threshold): rest, no exertion, hydrate.
        * **Hypoglycaemia** (glucose ≤ threshold): 15 g fast-acting carb,
          re-check in 15 minutes.
        * **Hypoxia** (SpO₂ ≤ 93%): sit upright, pursed-lip breathing.
        * **Bradycardia / tachycardia**: rest, no stimulants.
        * **Fallback**: rest and seek assistance if symptoms worsen.

        Args:
            vitals_dict: Flat vitals dict with numeric values.

        Returns:
            Ordered list of plain-English guidance strings.
        """
        guidance: list[str] = []
        sbp = vitals_dict.get("sbp")
        glucose = vitals_dict.get("glucose_mgdl")
        spo2 = vitals_dict.get("spo2")
        hr = vitals_dict.get("heart_rate")

        if sbp is not None and sbp >= HYPERTENSION_SELF_CARE_THRESHOLD:
            guidance.extend(
                [
                    "Sit or lie down quietly in a comfortable position.",
                    "Avoid physical exertion or stressful activities.",
                    "Stay hydrated — sip water slowly.",
                    "Take your blood pressure medication if it is due.",
                ]
            )

        if glucose is not None and glucose <= HYPOGLYCAEMIA_SELF_CARE_THRESHOLD:
            guidance.extend(
                [
                    "Consume 15 g of fast-acting carbohydrate immediately "
                    "(e.g. 4 glucose tablets, 150 mL fruit juice).",
                    "Wait 15 minutes, then re-check your blood glucose.",
                    "If still below 70 mg/dL, repeat the 15 g carbohydrate.",
                    "Do not drive or operate machinery until fully recovered.",
                ]
            )

        if spo2 is not None and spo2 <= 93.0:
            guidance.extend(
                [
                    "Sit upright and lean slightly forward.",
                    "Practise pursed-lip breathing: inhale slowly, exhale through "
                    "pursed lips for twice as long.",
                    "Avoid exertion — stay still and conserve oxygen.",
                ]
            )

        if hr is not None:
            if hr < 50:
                guidance.append(
                    "Your heart rate is low — sit down, stay warm, and avoid "
                    "stimulants such as caffeine."
                )
            elif hr > 120:
                guidance.append(
                    "Your heart rate is elevated — sit quietly, practise slow "
                    "diaphragmatic breathing, and avoid stimulants."
                )

        if not guidance:
            guidance = [
                "Rest in a comfortable position.",
                "Monitor your readings every 5 minutes.",
                "Contact your care team or call emergency services "
                "if symptoms worsen.",
            ]
        return guidance

    def clear_patient_state(self, patient_id: str) -> None:
        """Remove a patient's repeat-measurement state from memory.

        Should be called by the orchestrator on patient discharge or
        explicit de-escalation.

        Args:
            patient_id: The pseudonymised patient identifier to clear.
        """
        self._pending_repeats.pop(patient_id, None)
        self._log.info("Cleared repeat state for patient %s.", patient_id)

    def pending_patients(self) -> list[str]:
        """Return the list of patient IDs currently in watch-and-repeat.

        Returns:
            List of patient pseudonym strings with active repeat schedules.
        """
        return list(self._pending_repeats.keys())
