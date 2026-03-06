"""
lifestyle_agent.py
==================
Listing 4 from the paper — Sleep & Lifestyle Agent.

Responsible for:
    * Adjusting recommended sleep window to patient chronotype and sleep debt.
    * Issuing caffeine-cutoff hygiene reminders when consumed after 14:00.
    * Prompting post-meal / sedentary-break activity.
    * Never making disruptive schedule shifts (max 15 min per adjustment).

**Design principle — light-touch nudges:**
    All recommendations are *prompts*, never mandates.  The agent discourages
    disruptive shifts: if sleep debt is < 30 min the bedtime is left unchanged.
    Walk prompts are only issued once per ``WALK_COOLDOWN_MINUTES`` window to
    avoid notification fatigue.

Architecture position::

    preprocessing → envs → agents → orchestrator
                                     ↑
                             lifestyle_agent.py (here)

Imports::

    from paai_healthcare.agents.lifestyle_agent import LifestyleAgent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import (
    ActionType,
    AgentAction,
    AgentResult,
    BaseAgent,
    PatientState,
    Urgency,
)

logger = logging.getLogger(__name__)

# Minimum minutes between successive walk prompts (prevents alarm fatigue)
WALK_COOLDOWN_MINUTES: int = 90

# Minimum sleep-debt (hours) before an adjustment recommendation is issued
SLEEP_DEBT_THRESHOLD_HOURS: float = 0.5

# Maximum single-session bedtime shift (minutes) to avoid circadian disruption
MAX_BEDTIME_SHIFT_MINUTES: int = 15


# ──────────────────────────────────────────────────────────────────────────────
# Domain-specific data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChronotypeWindow:
    """Recommended bed/wake times for a patient chronotype.

    Attributes:
        chronotype:  ``"morning"``, ``"intermediate"``, or ``"evening"``.
        bedtime:     HH:MM 24-h recommended bedtime.
        wake_time:   HH:MM 24-h recommended wake time.
    """

    chronotype: str
    bedtime: str
    wake_time: str


# Canonical chronotype windows (evidence-based; Roenneberg 2012)
_CHRONOTYPE_TABLE: dict[str, ChronotypeWindow] = {
    "morning": ChronotypeWindow("morning", "21:30", "05:30"),
    "intermediate": ChronotypeWindow("intermediate", "22:30", "06:30"),
    "evening": ChronotypeWindow("evening", "23:30", "07:30"),
}
_DEFAULT_WINDOW = _CHRONOTYPE_TABLE["intermediate"]


@dataclass
class LifestylePlan:
    """Structured output of the Lifestyle Agent for one timestep.

    Attributes:
        recommended_bedtime:  Adjusted HH:MM bedtime (may differ from base
                              chronotype window if sleep debt is present).
        recommended_wake:     HH:MM wake time (chronotype-based; not adjusted).
        sleep_debt_hours:     Quantified sleep deficit at time of planning.
        prompts:              Ordered list of nudge dicts issued this step.
    """

    recommended_bedtime: str
    recommended_wake: str
    sleep_debt_hours: float
    prompts: list[dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# LifestyleAgent
# ──────────────────────────────────────────────────────────────────────────────


class LifestyleAgent(BaseAgent):
    """BDI Sleep & Lifestyle Agent — chronotype scheduling and hygiene nudges.

    **BDI mental state:**

    * Beliefs  — sleep efficiency, chronotype, steps today, hour of day,
                 caffeine intake, conditions, glucose.
    * Desires  — optimise sleep quality, reduce sedentary time, safe
                 caffeine use, maintain chronotype-aligned schedule.
    * Intentions — sleep_advance, nap_recommendation, caffeine_hygiene,
                   walk_prompt.

    **Key decision rules (Listing 4):**

    1. Compute sleep debt = max(0, target − actual) hours.
    2. If debt > :data:`SLEEP_DEBT_THRESHOLD_HOURS`:
       recommend moving bedtime earlier by :data:`MAX_BEDTIME_SHIFT_MINUTES`.
    3. If debt > 1.5 h and no nap today: recommend a short nap.
    4. If ``hour_of_day ≥ caffeine_cutoff`` and caffeine consumed: issue hygiene tip.
    5. If ``steps_today < 3000`` OR
       ``(glucose > 160 AND steps_today < 5000)``: issue walk prompt
       (subject to :data:`WALK_COOLDOWN_MINUTES` cool-down).

    Args:
        policy_registry:       Registry — provides caffeine restriction rules.
        knowledge_graph:       Clinical KG (not used directly; available for
                               future condition-specific lifestyle rules).
        audit_log:             Hash-chained audit sink.
        walk_cooldown_minutes: Minutes between walk prompts (default 90).
    """

    _DEFAULT_DESIRES = [
        "optimise_sleep_quality",
        "maintain_chronotype_schedule",
        "reduce_sedentary_streak",
        "enforce_caffeine_cutoff",
    ]

    def __init__(
        self,
        policy_registry: Any,
        knowledge_graph: Any,
        audit_log: Any,
        walk_cooldown_minutes: int = WALK_COOLDOWN_MINUTES,
    ) -> None:
        super().__init__(
            agent_id="lifestyle_agent",
            policy_registry=policy_registry,
            knowledge_graph=knowledge_graph,
            audit_log=audit_log,
        )
        self._walk_cooldown_minutes = walk_cooldown_minutes
        # Tracks minutes since last walk prompt per patient (in-memory state)
        self._last_walk_prompt_min: dict[str, float] = {}

    # ── BDI cycle ─────────────────────────────────────────────────────────────

    def perceive(self, state: PatientState) -> None:
        """Ingest patient state and populate lifestyle-domain beliefs.

        Beliefs populated:

        * ``patient_id``, ``conditions``
        * ``sleep_actual_hours``, ``sleep_target_hours`` — sleep debt calc
        * ``chronotype``         — one of ``morning`` / ``intermediate`` /
                                   ``evening``
        * ``steps_today``        — pedometer reading
        * ``hour_of_day``        — local hour (0–23)
        * ``caffeine_intake_mg`` — today's caffeine so far
        * ``glucose``            — post-prandial glucose for walk decision
        * ``nap_taken_today``    — bool from ``state.extra``
        * ``minutes_since_last_walk_prompt`` — for cool-down tracking

        Args:
            state: Current patient state from the orchestrator.
        """
        pid = state.patient_id
        self.beliefs.update(
            {
                "patient_id": pid,
                "conditions": state.conditions,
                "sleep_actual_hours": state.sleep_actual_hours,
                "sleep_target_hours": state.sleep_target_hours,
                "chronotype": state.chronotype,
                "steps_today": state.steps_today,
                "hour_of_day": state.hour_of_day,
                "caffeine_intake_mg": state.caffeine_intake_mg,
                "glucose": (state.vitals.glucose_mgdl or 100.0),
                "nap_taken_today": state.extra.get("nap_taken_today", False),
                "minutes_since_last_walk_prompt": (
                    self._last_walk_prompt_min.get(pid, 9999)
                ),
            }
        )
        self._log.debug(
            "Perceived: patient=%s sleep=%.1f/%.1f h steps=%d hour=%d",
            pid,
            state.sleep_actual_hours,
            state.sleep_target_hours,
            state.steps_today,
            state.hour_of_day,
        )

    def deliberate(self) -> list[dict]:
        """Derive lifestyle intentions from current beliefs.

        Checks (in order):

        1. Sleep debt → ``sleep_advance`` or ``nap_recommendation``.
        2. Caffeine after cutoff → ``caffeine_hygiene``.
        3. Low steps / high glucose → ``walk_prompt``.

        Returns:
            List of intention dicts (may be empty if all targets are met).
        """
        intentions: list[dict] = []

        # ── 1. Sleep ──────────────────────────────────────────────────────────
        debt = self._sleep_debt_hours()
        if debt >= SLEEP_DEBT_THRESHOLD_HOURS:
            intentions.append(
                {
                    "type": ActionType.SLEEP_ADJUSTMENT.value,  # was SLEEP_ADVANCE (Issue 2)
                    "urgency": Urgency.GENTLE.value,
                    "debt_h": debt,
                }
            )
        if debt >= 1.5 and not self.beliefs.get("nap_taken_today", False):
            intentions.append(
                {
                    "type": ActionType.NAP_RECOMMENDATION.value,
                    "urgency": Urgency.GENTLE.value,
                    "debt_h": debt,
                }
            )

        # ── 2. Caffeine cutoff ────────────────────────────────────────────────
        hour = int(self.beliefs.get("hour_of_day", 12))
        cutoff = self._caffeine_cutoff_hour()
        caffeine = float(self.beliefs.get("caffeine_intake_mg", 0.0))
        if hour >= cutoff and caffeine > 0:
            intentions.append(
                {
                    "type": ActionType.CAFFEINE_HYGIENE.value,
                    "urgency": Urgency.GENTLE.value,
                    "hour": hour,
                    "cutoff": cutoff,
                }
            )

        # ── 3. Activity ───────────────────────────────────────────────────────
        if self._should_prompt_walk():
            intentions.append(
                {
                    "type": ActionType.WALK_PROMPT.value,
                    "urgency": Urgency.GENTLE.value,
                    "steps": self.beliefs.get("steps_today", 0),
                    "glucose": self.beliefs.get("glucose", 100.0),
                }
            )

        self._log.debug("Deliberated %d intentions", len(intentions))
        return intentions

    def act(self, intentions: list[dict]) -> AgentResult:
        """Convert lifestyle intentions into typed :class:`AgentAction` objects.

        Assembles a :class:`LifestylePlan` and packages it alongside
        individual nudge actions.

        Args:
            intentions: List of intention dicts from :meth:`deliberate`.

        Returns:
            :class:`AgentResult` with per-intention actions and a plan
            summary in ``metadata``.
        """
        window = self._chronotype_window()
        debt = self._sleep_debt_hours()
        plan = LifestylePlan(
            recommended_bedtime=window.bedtime,
            recommended_wake=window.wake_time,
            sleep_debt_hours=debt,
        )

        actions: list[AgentAction] = []

        for intent in intentions:
            intent_type = intent.get("type", "")

            # ── Sleep advance ─────────────────────────────────────────────────
            if (
                intent_type == ActionType.SLEEP_ADJUSTMENT.value
            ):  # was SLEEP_ADVANCE (Issue 2)
                adjusted_bedtime = self._advance_bedtime(
                    window.bedtime, MAX_BEDTIME_SHIFT_MINUTES
                )
                plan.recommended_bedtime = adjusted_bedtime
                plan.prompts.append(
                    {
                        "type": "sleep_adjustment",
                        "adjusted_bedtime": adjusted_bedtime,
                        "shift_minutes": MAX_BEDTIME_SHIFT_MINUTES,
                    }
                )
                action = self._make_action(
                    action_type=ActionType.SLEEP_ADJUSTMENT,  # was SLEEP_ADVANCE (Issue 2)
                    urgency=Urgency.GENTLE,
                    payload={
                        "sleep_debt_hours": round(debt, 2),
                        "adjusted_bedtime": adjusted_bedtime,
                        "shift_minutes": MAX_BEDTIME_SHIFT_MINUTES,
                        "message": (
                            f"You have {debt:.1f} h of sleep debt.  "
                            f"Try moving bedtime "
                            f"{MAX_BEDTIME_SHIFT_MINUTES} min earlier "
                            f"to {adjusted_bedtime}."
                        ),
                    },
                    rationale=(
                        f"Sleep debt {debt:.2f} h ≥ threshold "
                        f"{SLEEP_DEBT_THRESHOLD_HOURS} h; light bedtime advance."
                    ),
                )
                actions.append(action)

            # ── Nap recommendation ────────────────────────────────────────────
            elif intent_type == ActionType.NAP_RECOMMENDATION.value:
                plan.prompts.append({"type": "nap_recommendation"})
                action = self._make_action(
                    action_type=ActionType.NAP_RECOMMENDATION,
                    urgency=Urgency.GENTLE,
                    payload={
                        "sleep_debt_hours": round(debt, 2),
                        "message": (
                            f"Your sleep debt is {debt:.1f} h.  "
                            "A 20-minute nap before 15:00 can help — "
                            "set an alarm to avoid oversleeping."
                        ),
                        "max_nap_minutes": 20,
                        "nap_before_hour": 15,
                    },
                    rationale=(f"Sleep debt {debt:.2f} h ≥ 1.5 h and no nap today."),
                )
                actions.append(action)

            # ── Caffeine hygiene ──────────────────────────────────────────────
            elif intent_type == ActionType.CAFFEINE_HYGIENE.value:
                cutoff = intent.get("cutoff", 14)
                plan.prompts.append({"type": "caffeine_hygiene", "cutoff": cutoff})
                action = self._make_action(
                    action_type=ActionType.CAFFEINE_HYGIENE,
                    urgency=Urgency.GENTLE,
                    payload={
                        "caffeine_intake_mg": self.beliefs.get(
                            "caffeine_intake_mg", 0.0
                        ),
                        "cutoff_hour": cutoff,
                        "message": (
                            f"Avoid caffeine after {cutoff:02d}:00 — "
                            "it can delay sleep onset by up to 40 minutes."
                        ),
                    },
                    rationale=(
                        f"Caffeine consumed at hour {intent.get('hour', 0)}, "
                        f"after the {cutoff}:00 cutoff."
                    ),
                )
                actions.append(action)

            # ── Walk prompt ───────────────────────────────────────────────────
            elif intent_type == ActionType.WALK_PROMPT.value:
                steps = intent.get("steps", 0)
                glucose = intent.get("glucose", 100.0)
                reason = (
                    "post-prandial glucose elevated"
                    if glucose > 160
                    else "low step count"
                )
                plan.prompts.append({"type": "walk_prompt", "reason": reason})
                action = self._make_action(
                    action_type=ActionType.WALK_PROMPT,
                    urgency=Urgency.GENTLE,
                    payload={
                        "steps_today": steps,
                        "glucose_mgdl": glucose,
                        "duration_min": 15,
                        "message": (
                            "A brisk 15-minute walk can lower blood glucose "
                            "and contribute to your daily step target.  "
                            "Even a short walk after meals helps."
                        ),
                    },
                    rationale=(
                        f"Walk prompted due to {reason} "
                        f"(steps={steps}, glucose={glucose:.0f} mg/dL)."
                    ),
                )
                actions.append(action)
                # Update cool-down tracker
                pid = str(self.beliefs.get("patient_id", ""))
                self._last_walk_prompt_min[pid] = 0.0

        self._persist_actions(actions)
        return AgentResult(
            agent_id=self.agent_id,
            actions=actions,
            metadata={
                "lifestyle_plan": {
                    "recommended_bedtime": plan.recommended_bedtime,
                    "recommended_wake": plan.recommended_wake,
                    "sleep_debt_hours": round(plan.sleep_debt_hours, 2),
                    "prompts": plan.prompts,
                },
            },
        )

    # ── Legacy dict interface (Issue 1 fix) ──────────────────────────────────

    def execute(self, task: dict) -> dict:
        """Execute one lifestyle task via the legacy dict protocol.

        Called by ``task_router.route(task, agents)``.  Merges ``task``
        into beliefs, runs the full BDI pipeline, and returns a plain dict.

        Args:
            task: Plain task dict.  May contain ``steps_today``,
                  ``sleep_actual_hours``, ``hour_of_day``,
                  ``caffeine_intake_mg``, ``conditions``, etc.

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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sleep_debt_hours(self) -> float:
        """Compute sleep debt as ``max(0, target − actual)`` hours.

        Returns:
            Non-negative float representing accumulated sleep debt.
        """
        target = float(self.beliefs.get("sleep_target_hours", 7.5))
        actual = float(self.beliefs.get("sleep_actual_hours", 7.0))
        return max(0.0, target - actual)

    def _chronotype_window(self) -> ChronotypeWindow:
        """Look up the recommended bed/wake window for the current chronotype.

        Returns:
            :class:`ChronotypeWindow` matching the patient's chronotype, or
            the intermediate window as a safe default.
        """
        chron = self.beliefs.get("chronotype", "intermediate")
        return _CHRONOTYPE_TABLE.get(chron, _DEFAULT_WINDOW)

    def _caffeine_cutoff_hour(self) -> int:
        """Retrieve the caffeine cutoff hour via the PolicyRegistryProtocol.

        Issue 3 fix: the previous implementation used
        ``getattr(self.registry, "rules", {})`` which accesses a concrete
        attribute of ``PolicyRegistry`` not declared in
        ``PolicyRegistryProtocol``.  Against any protocol-conformant mock or
        stub the lookup silently fell back to hardcoded 14 every time.

        Now delegates to ``self.registry.get_caffeine_cutoff(condition)``
        which is declared in the protocol and implemented concretely in
        ``knowledge/policy_registry.py``.

        Returns:
            Integer hour (0–23) after which caffeine should be avoided.
            Falls back to 14 if the registry raises unexpectedly.
        """
        conditions = self.beliefs.get("conditions", [])
        condition_key = "hypertension" if "hypertension" in conditions else "healthy"
        try:
            return int(self.registry.get_caffeine_cutoff(condition_key))
        except Exception:
            return 14  # safe default

    def _should_prompt_walk(self) -> bool:
        """Determine whether a walk prompt should be issued now.

        Criteria:

        * ``steps_today < 3000`` (absolute low-activity threshold), OR
        * ``glucose > 160 AND steps_today < 5000`` (post-prandial intervention).

        AND cool-down period has elapsed since the last walk prompt.

        Returns:
            ``True`` if a walk prompt is appropriate.
        """
        steps = int(self.beliefs.get("steps_today", 0))
        glucose = float(self.beliefs.get("glucose", 100.0))
        since = float(self.beliefs.get("minutes_since_last_walk_prompt", 9999))

        activity_trigger = (steps < 3000) or (glucose > 160 and steps < 5000)
        cooldown_elapsed = since >= self._walk_cooldown_minutes
        return activity_trigger and cooldown_elapsed

    @staticmethod
    def _advance_bedtime(bedtime_hhmm: str, minutes: int) -> str:
        """Shift a HH:MM bedtime string earlier by *minutes* minutes.

        Handles midnight wrap-around correctly.

        Args:
            bedtime_hhmm: Bedtime string in ``"HH:MM"`` format.
            minutes:      Number of minutes to advance (positive = earlier).

        Returns:
            New bedtime string in ``"HH:MM"`` format.

        Example::

            >>> LifestyleAgent._advance_bedtime("22:30", 15)
            "22:15"
            >>> LifestyleAgent._advance_bedtime("00:10", 15)
            "23:55"
        """
        try:
            h, m = map(int, bedtime_hhmm.split(":"))
        except ValueError:
            return bedtime_hhmm  # Return unchanged if parsing fails

        total_minutes = h * 60 + m - minutes
        total_minutes %= 24 * 60  # Wrap around midnight

        new_h, new_m = divmod(total_minutes, 60)
        return f"{new_h:02d}:{new_m:02d}"
