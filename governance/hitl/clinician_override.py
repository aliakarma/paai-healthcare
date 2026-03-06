"""
clinician_override.py
=====================
Tier 2 HiTL: Clinician override logging.
Overrides are stored as 5-tuples and fed back to offline policy update
to prevent RL from repeating rejected actions in similar states.
"""

from governance.audit_log import AuditLog


class ClinicianOverrideLogger:
    """Tier 2 override tuple: <s_t, a_proposed, a_override, clinician_id, rationale>"""

    def __init__(self, audit_log: AuditLog):
        self.audit = audit_log
        self._overrides: list[dict] = []

    def log_override(
        self,
        state: dict,
        proposed_action: int,
        override_action: int,
        clinician_id: str,
        rationale: str,
    ) -> dict:
        override = {
            "state": state,
            "action_proposed": proposed_action,
            "action_override": override_action,
            "clinician_id": clinician_id,
            "rationale": rationale,
        }
        self._overrides.append(override)
        self.audit.append(
            str(state.get("patient_id", "unknown")),
            f"clinician:{clinician_id}",
            "tier2_override",
            override,
        )
        return override

    def get_overrides_for_training(self) -> list[dict]:
        """Retrieve overrides for offline policy update."""
        return self._overrides.copy()
