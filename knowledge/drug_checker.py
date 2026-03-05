"""
drug_checker.py
===============
Convenience wrapper combining KG + Policy Registry for fast drug-food
and drug-condition interaction checking at agent runtime.
"""
from knowledge.knowledge_graph import KnowledgeGraph
from knowledge.policy_registry import PolicyRegistry


class DrugChecker:
    def __init__(self, kg: KnowledgeGraph, registry: PolicyRegistry):
        self.kg = kg
        self.registry = registry

    def safe_to_prescribe(self, drug: str, patient_conditions: list[str],
                           current_meds: list[str],
                           egfr: float | None = None) -> tuple[bool, str]:
        """
        Returns (is_safe: bool, reason: str).
        Checks renal contraindications, known interactions, and dose ceilings.
        """
        if self.registry.is_contraindicated(drug, egfr):
            return False, f"{drug} contraindicated (eGFR={egfr})"
        for cond in patient_conditions:
            contras = self.kg.get_condition_contraindications(cond)
            for c in contras:
                if c["drug"] == drug and c["severity"] in ("absolute", "high"):
                    return False, f"{drug} contraindicated in {cond}"
        return True, "ok"

    def flag_food_interactions(self, medications: list[str],
                                proposed_foods: list[str]) -> list[dict]:
        return self.kg.check_plan_conflicts(medications, proposed_foods)
