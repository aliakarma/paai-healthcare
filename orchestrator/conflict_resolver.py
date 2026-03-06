"""
conflict_resolver.py
====================
Resolves cross-domain conflicts in the proposed task set.
E.g., a late-night snack proposed by the nutrition agent
conflicts with the sodium budget and elevated BP.
"""


class ConflictResolver:
    """
    Checks proposed tasks for cross-domain inconsistencies.
    Implements the Conflict Resolver block (Figure 2, paper).
    """

    def __init__(self, knowledge_graph, policy_registry):
        self.kg = knowledge_graph
        self.registry = policy_registry

    def resolve(
        self, tasks: list[dict], vitals: dict, running_sodium_mg: float = 0.0
    ) -> list[dict]:
        """
        Resolve conflicts in the task list.
        Returns the cleaned (consistent) task list.
        """
        resolved = []
        for task in tasks:
            conflict, reason = self._check_task(task, vitals, running_sodium_mg)
            if not conflict:
                resolved.append(task)
            else:
                # Replace conflicting task with a safe alternative
                resolved.append(
                    {
                        "type": "conflict_replacement",
                        "original_task": task,
                        "reason": reason,
                        "safe_alternative": self._safe_alternative(task, reason),
                    }
                )
        return resolved

    def _check_task(
        self, task: dict, vitals: dict, running_sodium_mg: float
    ) -> tuple[bool, str]:
        t = task.get("type", "")
        sbp = vitals.get("sbp", 120)

        # Sodium budget enforcement
        if t == "dietary_modification":
            sodium = task.get("sodium_mg", 0)
            cap = self.registry.get_sodium_cap("hypertension") * 1000
            if running_sodium_mg + sodium > cap:
                return (
                    True,
                    f"sodium_budget_exceeded ({running_sodium_mg + sodium:.0f}mg)",
                )

        # No late snacks when BP is high
        if t == "snack_recommendation" and sbp > 160:
            return True, f"snack_conflict_high_bp ({sbp}mmHg)"

        # Potassium-rich food for CKD patient
        if t == "dietary_modification":
            if task.get("potassium_rich") and self.registry.rules.get(
                "renal_adjustments"
            ):
                return True, "potassium_restricted_ckd"

        return False, "ok"

    def _safe_alternative(self, task: dict, reason: str) -> dict:
        if "sodium" in reason:
            return {
                "type": "sodium_advisory",
                "message": "Daily sodium budget reached. Choose unseasoned foods.",
            }
        if "snack" in reason:
            return {
                "type": "hydration_prompt",
                "message": "Sip water instead of a snack to help manage blood pressure.",
            }
        return {"type": "generic_safe", "message": "Please follow your care plan."}
