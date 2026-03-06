"""Unit tests for governance and HiTL modules."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from governance.audit_log import AuditLog
from governance.hitl.clinician_override import ClinicianOverrideLogger
from governance.hitl.patient_feedback import PatientFeedbackCollector, FEEDBACK_REWARD_MAP


class TestAuditLog:
    """Test immutable audit log functionality."""

    def test_audit_log_initialization(self):
        """Test that audit log initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_audit.jsonl")
            audit = AuditLog(log_path=log_path)
            
            assert audit.log_path == Path(log_path)
            assert audit._entry_count == 0
            assert audit._last_hash == "GENESIS"

    def test_audit_log_append_entry(self):
        """Test appending entries to audit log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_audit.jsonl")
            audit = AuditLog(log_path=log_path)
            
            audit.append("patient_001", "system", "test_event", {"data": "test"})
            
            assert audit._entry_count == 1
            assert Path(log_path).exists()

    def test_audit_log_persistence(self):
        """Test that audit log persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_audit.jsonl")
            
            # Create and append
            audit1 = AuditLog(log_path=log_path)
            audit1.append("patient_001", "system", "event_1", {"data": "test1"})
            
            # Create new instance and reload
            audit2 = AuditLog(log_path=log_path)
            
            assert audit2._entry_count == 1
            assert audit2._last_hash != "GENESIS"

    def test_audit_log_hash_chain(self):
        """Test that entries form a hash chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_audit.jsonl")
            audit = AuditLog(log_path=log_path)
            
            # Append multiple entries
            audit.append("patient_001", "system", "event_1", {"data": "test1"})
            hash1 = audit._last_hash
            
            audit.append("patient_001", "system", "event_2", {"data": "test2"})
            hash2 = audit._last_hash
            
            assert hash1 != hash2, "Each entry should have different hash"
            assert hash1 != "GENESIS", "First entry hash should differ from genesis"

    def test_audit_log_verify_integrity(self):
        """Test integrity verification of audit log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_audit.jsonl")
            audit1 = AuditLog(log_path=log_path)
            
            # Append several entries
            audit1.append("patient_001", "system", "event_1", {"data": "test1"})
            audit1.append("patient_001", "system", "event_2", {"data": "test2"})
            audit1.append("patient_001", "system", "event_3", {"data": "test3"})
            
            # Reload and verify
            audit2 = AuditLog(log_path=log_path)
            is_valid = audit2.verify_integrity()
            
            assert is_valid == True, "Unmodified audit log should pass integrity check"

    def test_audit_log_tampering_detection(self):
        """Test that tampering is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_audit.jsonl")
            audit = AuditLog(log_path=log_path)
            
            # Append entries
            audit.append("patient_001", "system", "event_1", {"data": "test1"})
            audit.append("patient_001", "system", "event_2", {"data": "test2"})
            
            # Tamper with the log file
            with open(log_path, "r") as f:
                lines = f.readlines()
            
            # Modify first entry
            first_entry = json.loads(lines[0])
            first_entry["data"]["tampered"] = True
            lines[0] = json.dumps(first_entry) + "\n"
            
            with open(log_path, "w") as f:
                f.writelines(lines)
            
            # Reload and verify
            audit2 = AuditLog(log_path=log_path)
            is_valid = audit2.verify_integrity()
            
            assert is_valid == False, "Tampered audit log should fail integrity check"

    def test_audit_log_multiple_patients(self):
        """Test audit log with multiple patients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_audit.jsonl")
            audit = AuditLog(log_path=log_path)
            
            # Append entries for different patients
            for pid in ["patient_001", "patient_002", "patient_003"]:
                audit.append(pid, "system", "event", {"data": f"for {pid}"})
            
            assert audit._entry_count == 3


class TestPatientFeedbackCollector:
    """Test patient feedback collection and reward mapping."""

    def test_feedback_collector_initialization(self):
        """Test feedback collector initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            collector = PatientFeedbackCollector(audit)
            
            assert collector._buffer == []

    def test_feedback_reward_mapping(self):
        """Test that feedback maps to correct rewards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            collector = PatientFeedbackCollector(audit)
            
            # Test each feedback type
            reward_accept = collector.record("p001", "rec_001", "accept")
            reward_modify = collector.record("p001", "rec_002", "modify")
            reward_reject = collector.record("p001", "rec_003", "reject")
            
            assert reward_accept == 1.0, "Accept should map to +1.0"
            assert reward_modify == 0.0, "Modify should map to 0.0"
            assert reward_reject == -1.0, "Reject should map to -1.0"

    def test_feedback_buffer_accumulation(self):
        """Test that feedback accumulates in buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            collector = PatientFeedbackCollector(audit)
            
            # Record multiple feedback items
            for i in range(5):
                collector.record(f"p{i:03d}", f"rec_{i}", "accept")
            
            assert len(collector._buffer) == 5

    def test_feedback_flush_to_rl(self):
        """Test flushing feedback buffer for RL update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            collector = PatientFeedbackCollector(audit)
            
            # Record feedback
            collector.record("p001", "rec_001", "accept")
            collector.record("p001", "rec_002", "reject")
            
            assert len(collector._buffer) == 2
            
            # Flush
            flushed = collector.flush_to_rl()
            
            assert len(flushed) == 2, "Should flush all buffered items"
            assert len(collector._buffer) == 0, "Buffer should be empty after flush"
            assert flushed[0]["feedback"] == "accept"
            assert flushed[1]["feedback"] == "reject"

    def test_feedback_modification_tracking(self):
        """Test that patient modifications are tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            collector = PatientFeedbackCollector(audit)
            
            modification = "Changed dose from 10mg to 5mg"
            collector.record("p001", "rec_001", "modify", modification=modification)
            
            flushed = collector.flush_to_rl()
            
            assert flushed[0]["modification"] == modification

    def test_feedback_audit_trail(self):
        """Test that feedback is recorded in audit log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            collector = PatientFeedbackCollector(audit)
            
            collector.record("p001", "rec_001", "accept")
            collector.record("p002", "rec_002", "reject")
            
            # Check that audit log has entries
            assert audit._entry_count == 2


class TestClinicianOverrideLogger:
    """Test clinician override logging for Tier 2 HiTL."""

    def test_override_logger_initialization(self):
        """Test override logger initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            logger = ClinicianOverrideLogger(audit)
            
            assert logger._overrides == []

    def test_override_logging(self):
        """Test logging a clinician override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            logger = ClinicianOverrideLogger(audit)
            
            state = {"patient_id": "p001", "sbp": 180}
            override = logger.log_override(
                state=state,
                proposed_action=3,
                override_action=4,
                clinician_id="doc_001",
                rationale="Patient showing signs of hypertensive crisis",
            )
            
            assert override["action_proposed"] == 3
            assert override["action_override"] == 4
            assert override["clinician_id"] == "doc_001"
            assert len(logger._overrides) == 1

    def test_override_tracking_multiple(self):
        """Test tracking multiple overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            logger = ClinicianOverrideLogger(audit)
            
            # Log multiple overrides
            for i in range(3):
                logger.log_override(
                    state={"patient_id": f"p{i:03d}", "sbp": 150 + i*10},
                    proposed_action=i % 5,
                    override_action=(i + 1) % 5,
                    clinician_id=f"doc_{i:03d}",
                    rationale=f"Override reason {i}",
                )
            
            assert len(logger._overrides) == 3

    def test_get_overrides_for_training(self):
        """Test retrieving overrides for offline policy update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            logger = ClinicianOverrideLogger(audit)
            
            # Log overrides
            logger.log_override(
                state={"patient_id": "p001"},
                proposed_action=1,
                override_action=4,
                clinician_id="doc_001",
                rationale="Severe hypoglycemia detected",
            )
            logger.log_override(
                state={"patient_id": "p002"},
                proposed_action=2,
                override_action=0,
                clinician_id="doc_002",
                rationale="Patient preference",
            )
            
            overrides = logger.get_overrides_for_training()
            
            assert len(overrides) == 2
            assert overrides[0]["action_override"] == 4
            assert overrides[1]["action_override"] == 0

    def test_override_audit_trail(self):
        """Test that overrides are recorded in audit log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            logger = ClinicianOverrideLogger(audit)
            
            logger.log_override(
                state={"patient_id": "p001"},
                proposed_action=1,
                override_action=4,
                clinician_id="doc_001",
                rationale="Emergency escalation",
            )
            
            assert audit._entry_count == 1

    def test_override_data_structure(self):
        """Test that override data is properly structured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            logger = ClinicianOverrideLogger(audit)
            
            state = {"patient_id": "p001", "sbp": 200, "glucose": 300}
            override = logger.log_override(
                state=state,
                proposed_action=3,
                override_action=4,
                clinician_id="doc_001",
                rationale="Multiple critical thresholds exceeded",
            )
            
            # Check structure
            assert "state" in override
            assert "action_proposed" in override
            assert "action_override" in override
            assert "clinician_id" in override
            assert "rationale" in override
            
            # Verify values
            assert override["state"] == state
            assert override["action_proposed"] == 3
            assert override["action_override"] == 4


class TestHiTLIntegration:
    """Integration tests for complete HiTL governance loop."""

    def test_patient_feedback_to_override_flow(self):
        """Test flow from patient feedback to clinician override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            
            # Patient rejects recommendation
            feedback_collector = PatientFeedbackCollector(audit)
            reward = feedback_collector.record("p001", "rec_001", "reject", 
                                             modification="Patient prefers rest over medication")
            
            # Clinician reviews and overrides
            override_logger = ClinicianOverrideLogger(audit)
            override = override_logger.log_override(
                state={"patient_id": "p001", "sbp": 160},
                proposed_action=1,
                override_action=3,  # Lifestyle modification instead
                clinician_id="doc_001",
                rationale="Respecting patient preference, trying lifestyle first",
            )
            
            # Verify flow
            assert reward == -1.0, "Rejection should give negative reward"
            assert override["action_override"] == 3
            assert audit._entry_count == 2
            assert audit.verify_integrity()

    def test_audit_consistency_across_hitl_operations(self):
        """Test that audit log remains consistent through HiTL operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "audit.jsonl")
            audit = AuditLog(log_path=log_path)
            
            # Multiple HiTL operations
            feedback = PatientFeedbackCollector(audit)
            override = ClinicianOverrideLogger(audit)
            
            feedback.record("p001", "rec_001", "accept")
            override.log_override(
                state={"patient_id": "p002"},
                proposed_action=1,
                override_action=4,
                clinician_id="doc_001",
                rationale="Emergency",
            )
            feedback.record("p003", "rec_003", "modify")
            
            # Verify integrity after all operations
            assert audit._entry_count == 3
            assert audit.verify_integrity()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
