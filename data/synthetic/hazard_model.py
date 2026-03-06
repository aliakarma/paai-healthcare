"""
hazard_model.py
===============
Rare event generator for the synthetic patient cohort.
Uses a conditional hazard rate model based on recent vital signs
and comorbidity load to inject clinically realistic escalation events.
"""

import numpy as np


class HazardModel:
    """
    Generates rare but clinically significant events:
      - Hypertensive urgency (SBP >= 180)
      - Hypoglycaemic episode (glucose <= 54)
      - ER visit (composite)

    Events are conditioned on recent vital sign levels and
    have a persistence duration to simulate real clinical scenarios.
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._active_events: dict[int, dict] = {}  # patient_id -> active event

    def check_event(
        self, patient_id: int, day: float, sbp: float, glucose: float
    ) -> str | None:
        """
        Determine if a rare event occurs or persists at this timestep.
        Returns event label string or None.
        """
        # Check if an active event is still persisting
        if patient_id in self._active_events:
            ev = self._active_events[patient_id]
            if day < ev["end_day"]:
                return ev["type"]
            else:
                del self._active_events[patient_id]

        # Compute instantaneous hazard rates
        hyp_rate = self.cfg["hypertensive_urgency_rate_per_year"] / 365
        hypo_rate = self.cfg["hypoglycemic_event_rate_per_year"] / 365

        # Condition on current vitals (higher rates near threshold)
        if sbp > 160:
            hyp_rate *= 3.0
        if glucose < 80:
            hypo_rate *= 4.0
        if glucose < 60:
            hypo_rate *= 10.0

        # Sample events (Poisson process approximation)
        if self.rng.random() < hyp_rate:
            duration = self.cfg["event_persistence_hours"] / 24
            self._active_events[patient_id] = {
                "type": "hypertensive_urgency",
                "end_day": day + duration,
            }
            return "hypertensive_urgency"

        if self.rng.random() < hypo_rate:
            duration = self.cfg["event_persistence_hours"] / 24 * 0.5
            self._active_events[patient_id] = {
                "type": "hypoglycemic_episode",
                "end_day": day + duration,
            }
            return "hypoglycemic_episode"

        # ER visit (composite — rarer)
        er_rate = self.cfg["er_visit_rate_per_year"] / 365
        if self.rng.random() < er_rate:
            return "er_visit"

        return None
