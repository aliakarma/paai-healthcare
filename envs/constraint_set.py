"""constraint_set.py — Feasibility checker for the CMDP constraint set C."""

import numpy as np

from knowledge.policy_registry import PolicyRegistry


class ConstraintSet:
    def __init__(self, registry: PolicyRegistry):
        self.registry = registry

    def is_feasible(self, action: int, vitals: dict, adherence_med: float) -> bool:
        if action == 4:
            return self.registry.should_escalate(vitals) or self.registry.should_watch(
                vitals
            )
        if action == 1 and adherence_med > 0.92:
            return False
        return True

    def violation_rate(
        self, actions: np.ndarray, vitals_list: list, adherence_list: list
    ) -> float:
        violations = sum(
            not self.is_feasible(int(a), v, ad)
            for a, v, ad in zip(actions, vitals_list, adherence_list)
        )
        return violations / len(actions) if actions.size else 0.0
