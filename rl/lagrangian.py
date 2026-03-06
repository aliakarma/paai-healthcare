"""
lagrangian.py
=============
Lagrangian multiplier update for constraint enforcement.
Used to penalise constraint violations during RL training.
"""


class LagrangianUpdater:
    """Updates the Lagrangian multiplier lambda to enforce safety constraints."""

    def __init__(
        self,
        threshold: float = 0.05,
        lr: float = 0.01,
        lambda_init: float = 1.0,
        lambda_max: float = 10.0,
    ):
        self.threshold = threshold  # Max allowed violation rate
        self.lr = lr
        self.lambda_val = lambda_init
        self.lambda_max = lambda_max
        self._violation_rates = []

    def update(self, violation_rate: float) -> float:
        """
        Update lambda based on observed constraint violation rate.
        If violation_rate > threshold → increase lambda (more penalty).
        If violation_rate < threshold → decrease lambda (relax).
        """
        self._violation_rates.append(violation_rate)
        constraint_surplus = violation_rate - self.threshold
        self.lambda_val = max(
            0.0, min(self.lambda_max, self.lambda_val + self.lr * constraint_surplus)
        )
        return self.lambda_val

    @property
    def current_lambda(self) -> float:
        return self.lambda_val

    def mean_violation_rate(self) -> float:
        if not self._violation_rates:
            return 0.0
        return float(sum(self._violation_rates) / len(self._violation_rates))
