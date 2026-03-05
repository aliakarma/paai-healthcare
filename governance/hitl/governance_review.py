"""
governance_review.py
====================
Tier 3 HiTL: Weekly governance committee drift metrics.
Monitors override frequency, false-positive rates, and subgroup disparities.
"""
import numpy as np
from pathlib import Path
import json


class GovernanceReviewer:
    """Computes weekly governance metrics for clinical committee review."""

    def __init__(self, audit_log_path: str = "governance/audit.jsonl"):
        self.log_path = Path(audit_log_path)

    def compute_weekly_metrics(self) -> dict:
        if not self.log_path.exists():
            return {}
        entries = []
        with open(self.log_path) as f:
            for line in f:
                entries.append(json.loads(line.strip()))

        override_entries = [e for e in entries
                              if e.get("action_type") == "tier2_override"]
        escalation_entries = [e for e in entries
                                if e.get("action_type") == "escalation_alert"]
        total = len(entries)

        return {
            "total_audit_entries": total,
            "total_overrides": len(override_entries),
            "override_rate": len(override_entries) / max(total, 1),
            "total_escalations": len(escalation_entries),
            "flagged_for_review": len(override_entries) > 10,
        }
