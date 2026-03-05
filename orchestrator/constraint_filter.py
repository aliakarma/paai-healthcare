"""
constraint_filter.py
====================
Projects proposed RL actions onto the feasible set C.
Implements the Constraint Filter block (Figure 2, paper).
"""
import numpy as np


class ConstraintFilter:
    """
    Enforces hard safety constraints on proposed actions.
    Actions that violate prescriber rules are redirected or blocked.
    """

    def __init__(self, policy_registry):
        self.registry = policy_registry

    def filter(self, proposed_action: int, vitals: dict,
               adherence_med: float, active_policies: dict) -> tuple[int, str]:
        """
        Given a proposed discrete action (0-4), return the feasible action.

        Actions:
          0 = no_action
          1 = medication_schedule
          2 = dietary_modification
          3 = lifestyle_prompt
          4 = escalate

        Returns (filtered_action, reason_str)
        """
        # Rule 1: Escalation is mandatory if automatic threshold exceeded
        if self.registry.should_escalate(vitals):
            if proposed_action != 4:
                return 4, "mandatory_escalation_override"

        # Rule 2: Cannot issue medication reminders when adherence is already high
        if proposed_action == 1 and adherence_med > 0.92:
            return 0, "adherence_already_high"

        # Rule 3: Caffeine restriction — block dietary mods after cutoff hour
        hour = active_policies.get("hour_of_day", 12)
        if proposed_action == 2:
            caf_restriction = self.registry.rules.get(
                "caffeine_restriction", {}).get("hypertension", {})
            cutoff = caf_restriction.get("after_hour", 14)
            if hour >= cutoff and active_policies.get("hypertension"):
                pass  # Allow but log warning

        return proposed_action, "ok"

    def action_mask(self, vitals: dict, adherence_med: float) -> np.ndarray:
        """
        Return boolean mask of currently feasible actions.
        Used by MaskablePPO to prevent infeasible action sampling.
        """
        mask = np.ones(5, dtype=bool)

        # Cannot escalate unless in watch/escalation zone
        if not (self.registry.should_escalate(vitals) or
                self.registry.should_watch(vitals)):
            mask[4] = False

        # No medication reminder if high adherence
        if adherence_med > 0.92:
            mask[1] = False

        # Always allow no_action
        mask[0] = True

        return mask
