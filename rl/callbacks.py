"""Training callbacks for monitoring and logging."""

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    # stable-baselines3 is an optional training dependency.
    # When the package is absent the callback class is still importable
    # (e.g. for unit tests that import the module) but cannot be used.
    class BaseCallback:  # type: ignore[no-redef]
        """Stub used when stable-baselines3 is not installed."""

        def __init__(self, verbose: int = 0):
            self.verbose = verbose

        def _on_step(self) -> bool:
            raise ImportError(
                "stable-baselines3 is required to use TensorboardRewardCallback. "
                "Install it with: pip install stable-baselines3"
            )


class TensorboardRewardCallback(BaseCallback):
    """Logs detailed reward components to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
        if len(self._episode_rewards) >= 10:
            self.logger.record(
                "custom/mean_ep_reward", float(np.mean(self._episode_rewards[-10:]))
            )
            self.logger.record(
                "custom/min_ep_reward", float(np.min(self._episode_rewards[-10:]))
            )
        return True
