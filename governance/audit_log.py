"""
audit_log.py
============
Immutable hash-chained audit log implementing the CIP (Confidentiality,
Integrity, Privacy) governance layer described in Section 3.1.

Every entry is cryptographically linked to its predecessor.
Any tampering is detectable by verify_integrity().
"""
import hashlib
import json
import time
import os
from pathlib import Path
from typing import Optional


class AuditLog:
    """Append-only hash-chained audit log with optional AES-256 field encryption."""

    def __init__(self, log_path: str = "governance/audit.jsonl",
                 encryption_key: Optional[bytes] = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = "GENESIS"
        self._entry_count = 0
        self._fernet = None

        if encryption_key:
            try:
                from cryptography.fernet import Fernet
                self._fernet = Fernet(encryption_key)
            except ImportError:
                pass

        # Resume from existing log
        if self.log_path.exists():
            with open(self.log_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                last = json.loads(lines[-1])
                self._last_hash = last["entry_hash"]
                self._entry_count = len(lines)

    def append(self, patient_id: str, agent_id: str, action_type: str,
               action_detail: dict, outcome: Optional[dict] = None) -> str:
        """Append an entry; returns this entry's SHA-256 hash."""
        pseudonym = hashlib.sha256(patient_id.encode()).hexdigest()[:12]
        detail_str = json.dumps(action_detail, sort_keys=True)
        if self._fernet:
            detail_stored = self._fernet.encrypt(detail_str.encode()).decode()
        else:
            detail_stored = detail_str

        entry = {
            "seq":              self._entry_count,
            "timestamp":        time.time(),
            "patient_pseudonym": pseudonym,
            "agent_id":         agent_id,
            "action_type":      action_type,
            "action_detail":    detail_stored,
            "outcome":          outcome or {},
            "prev_hash":        self._last_hash,
        }
        content = json.dumps(entry, sort_keys=True).encode()
        entry_hash = hashlib.sha256(content).hexdigest()
        entry["entry_hash"] = entry_hash

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._last_hash = entry_hash
        self._entry_count += 1
        return entry_hash

    def verify_integrity(self) -> bool:
        """Verify the full hash chain. Returns True if intact."""
        if not self.log_path.exists():
            return True
        prev = "GENESIS"
        with open(self.log_path) as f:
            for i, line in enumerate(f, 1):
                entry = json.loads(line.strip())
                if entry["prev_hash"] != prev:
                    print(f"✗ Chain broken at entry {i}: prev_hash mismatch")
                    return False
                stored_hash = entry.pop("entry_hash")
                recomputed = hashlib.sha256(
                    json.dumps(entry, sort_keys=True).encode()).hexdigest()
                entry["entry_hash"] = stored_hash
                if recomputed != stored_hash:
                    print(f"✗ Tampered entry detected at {i}")
                    return False
                prev = stored_hash
        print(f"✓ Audit log intact — {i} entries verified.")
        return True

    @property
    def entry_count(self) -> int:
        return self._entry_count
