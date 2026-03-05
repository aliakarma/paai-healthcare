"""Unit tests for hash-chained audit log."""
import sys, os, tempfile, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_append_and_verify():
    from governance.audit_log import AuditLog
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        path = tf.name
    log = AuditLog(path)
    log.append("patient_001", "medicine_agent", "reminder",
                {"drug": "metformin"}, {"sent": True})
    log.append("patient_001", "nutrition_agent", "meal_plan",
                {"meal": "oatmeal"}, {})
    assert log.verify_integrity()
    os.unlink(path)


def test_tamper_detection():
    import json
    from governance.audit_log import AuditLog
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                      delete=False) as tf:
        path = tf.name
    log = AuditLog(path)
    log.append("p1", "agent", "test", {"x": 1})
    # Tamper: modify the file
    with open(path, "r") as f:
        lines = f.readlines()
    entry = json.loads(lines[0])
    entry["action_detail"] = '{"x": 999}'   # tamper
    with open(path, "w") as f:
        f.write(json.dumps(entry) + "\n")
    log2 = AuditLog(path)
    assert not log2.verify_integrity()
    os.unlink(path)


def test_entry_count():
    from governance.audit_log import AuditLog
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        path = tf.name
    log = AuditLog(path)
    for i in range(5):
        log.append(f"p{i}", "agent", "type", {"i": i})
    assert log.entry_count == 5
    os.unlink(path)
