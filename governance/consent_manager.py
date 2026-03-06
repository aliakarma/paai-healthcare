"""consent_manager.py — Tracks patient consent scopes per data channel."""

import json
from pathlib import Path


class ConsentManager:
    """
    Per-patient consent scope registry.
    Consent is required before sending any alert to a clinician (HIPAA/GDPR).
    """

    CHANNELS = [
        "clinician_alert",
        "cardiologist_share",
        "research_data",
        "family_notification",
    ]

    def __init__(self, store_path: str = "governance/consent_store.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if self.store_path.exists():
            self._store = json.loads(self.store_path.read_text())
        else:
            self._store = {}

    def grant(self, patient_id: str, channel: str):
        self._store.setdefault(patient_id, {})[channel] = True
        self._persist()

    def revoke(self, patient_id: str, channel: str):
        self._store.setdefault(patient_id, {})[channel] = False
        self._persist()

    def has_consent(self, patient_id: str, channel: str) -> bool:
        return bool(self._store.get(patient_id, {}).get(channel, False))

    def _persist(self):
        self.store_path.write_text(json.dumps(self._store, indent=2))
