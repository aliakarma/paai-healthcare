"""
patient_feedback.py
===================
Tier 1 HiTL: Patient accept/modify/reject feedback.
Feedback is mapped to R_t^adherence and fed into the RL reward signal.
"""
from typing import Literal
from governance.audit_log import AuditLog


FeedbackType = Literal["accept", "modify", "reject"]
FEEDBACK_REWARD_MAP = {"accept": 1.0, "modify": 0.0, "reject": -1.0}


class PatientFeedbackCollector:
    def __init__(self, audit_log: AuditLog):
        self.audit = audit_log
        self._buffer: list[dict] = []

    def record(self, patient_id: str, recommendation_id: str,
               feedback: FeedbackType, modification: str = ""):
        reward_signal = FEEDBACK_REWARD_MAP.get(feedback, 0.0)
        entry = {
            "patient_id": patient_id,
            "recommendation_id": recommendation_id,
            "feedback": feedback,
            "adherence_reward": reward_signal,
            "modification": modification,
        }
        self._buffer.append(entry)
        self.audit.append(patient_id, "patient_app", "tier1_feedback", entry)
        return reward_signal

    def flush_to_rl(self) -> list[dict]:
        """Return buffered feedback for RL offline update and clear buffer."""
        out = self._buffer.copy()
        self._buffer.clear()
        return out
