"""
policy_registry.py
==================
Loads prescriber rules, allergy exclusions, and escalation criteria.
Exposes a constraint-checking interface used by the Constraint Filter.
"""
import json
from pathlib import Path


class PolicyRegistry:
    """Rule-based constraint engine."""

    def __init__(self, registry_dir: str = "data/policy_registry/"):
        d = Path(registry_dir)
        self.rules = json.loads((d / "prescriber_rules.json").read_text())
        self.allergies = json.loads((d / "allergy_exclusions.json").read_text())
        self.escalation = json.loads((d / "escalation_criteria.json").read_text())

    def get_sodium_cap(self, condition: str) -> float:
        return self.rules["sodium_cap_g_per_day"].get(condition, 3.5)

    def get_timing_window(self, drug: str) -> dict | None:
        return self.rules["timing_windows"].get(drug)

    def get_dose_ceiling(self, drug_key: str) -> float | None:
        return self.rules["dose_ceilings"].get(drug_key)

    def get_caffeine_cutoff(self, condition: str) -> int:
        """Return the hour (0-23) after which caffeine should be avoided.

        Reads from rules['caffeine_restriction'][condition]['after_hour'].
        Falls back to 14 if the condition key or 'after_hour' is not present.

        Args:
            condition: Patient condition key, e.g. 'hypertension' or 'healthy'.

        Returns:
            Integer hour in [0, 23].
        """
        restriction = self.rules.get("caffeine_restriction", {})
        condition_rules = restriction.get(condition, restriction.get("healthy", {}))
        return int(condition_rules.get("after_hour", 14))

    def is_contraindicated(self, drug: str, egfr: float | None = None) -> bool:
        adj = self.rules.get("renal_adjustments", {})
        if drug in adj and egfr is not None:
            return egfr < adj[drug].get("hold_if_egfr_below", 0)
        return False

    def should_escalate(self, vitals: dict) -> bool:
        """Return True if any vital exceeds automatic escalation threshold."""
        auto = self.escalation["automatic_escalation"]
        sbp = vitals.get("sbp", 0)
        glc = vitals.get("glucose_mgdl", 100)
        hr  = vitals.get("heart_rate", 70)
        spo2 = vitals.get("spo2", 98)
        return (
            sbp >= auto.get("systolic_bp_mmhg_gte", 9999) or
            glc <= auto.get("glucose_mgdl_lte", -1) or
            glc >= auto.get("glucose_mgdl_gte", 9999) or
            spo2 <= auto.get("spo2_pct_lte", -1) or
            hr <= auto.get("heart_rate_bpm_lte", -1) or
            hr >= auto.get("heart_rate_bpm_gte", 9999)
        )

    def should_watch(self, vitals: dict) -> bool:
        """Return True if vitals are in watch-and-repeat zone."""
        watch = self.escalation["watch_and_repeat"]
        sbp = vitals.get("sbp", 0)
        glc = vitals.get("glucose_mgdl", 100)
        spo2 = vitals.get("spo2", 98)
        return (
            sbp >= watch.get("systolic_bp_mmhg_gte", 9999) or
            glc <= watch.get("glucose_mgdl_lte", -1) or
            glc >= watch.get("glucose_mgdl_gte", 9999) or
            spo2 <= watch.get("spo2_pct_lte", -1)
        )
