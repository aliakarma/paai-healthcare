"""
adherence_model.py
==================
Stochastic patient adherence model for medication and dietary compliance.
Uses a Markov-like process with autocorrelation to model realistic
adherence patterns including streaks and lapses.
"""
import numpy as np


class AdherenceModel:
    """
    Models patient adherence over time with:
      - Memory: today's adherence influenced by past N days
      - Dose miss / delay events (stochastic)
      - Medication and dietary adherence tracked separately
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._cache: dict[int, list] = {}

    def _initialise_patient(self, patient_id: int, n_days: int):
        """Pre-generate full adherence timeline for one patient."""
        mem = self.cfg["adherence_memory_days"]
        med_base = self.cfg["mean_medication_adherence"]
        diet_base = self.cfg["mean_dietary_adherence"]
        miss_p = self.cfg["dose_miss_prob"]
        delay_p = self.cfg["dose_delay_prob"]

        med_arr = np.zeros(n_days)
        diet_arr = np.zeros(n_days)

        for d in range(n_days):
            # Autocorrelated adherence: weighted mean of recent history
            if d == 0:
                med_arr[d] = self.rng.beta(
                    med_base * 10, (1 - med_base) * 10)
                diet_arr[d] = self.rng.beta(
                    diet_base * 10, (1 - diet_base) * 10)
            else:
                window = med_arr[max(0, d - mem):d]
                trend_med = np.mean(window) if len(window) else med_base
                trend_diet_w = diet_arr[max(0, d - mem):d]
                trend_diet = np.mean(trend_diet_w) if len(trend_diet_w) else diet_base

                # Stochastic perturbation
                med_today = np.clip(trend_med + self.rng.normal(0, 0.08), 0, 1)
                diet_today = np.clip(trend_diet + self.rng.normal(0, 0.1), 0, 1)

                # Dose miss event
                if self.rng.random() < miss_p:
                    med_today *= 0.0
                elif self.rng.random() < delay_p:
                    med_today *= 0.6

                med_arr[d] = med_today
                diet_arr[d] = diet_today

        self._cache[patient_id] = {
            "medication": med_arr,
            "dietary": diet_arr,
        }

    def get_adherence(self, patient_id: int, day: float,
                      total_days: int = 365) -> dict:
        """Return adherence scores for a given day (float)."""
        if patient_id not in self._cache:
            self._initialise_patient(patient_id, total_days)
        day_idx = min(int(day), total_days - 1)
        return {
            "medication": float(self._cache[patient_id]["medication"][day_idx]),
            "dietary": float(self._cache[patient_id]["dietary"][day_idx]),
            "lifestyle": float(
                self.rng.beta(5, 3)),  # lifestyle varies more day-to-day
        }
