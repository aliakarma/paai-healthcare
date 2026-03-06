"""
patient_env.py
==============
Custom OpenAI Gymnasium environment for AgHealth+.
The RL policy pi(a|s) is trained and evaluated within this environment.

State  : 25-dimensional vector (vitals + rolling stats + adherence + context + policies)
Actions: Discrete(5) — {no_action, med_schedule, dietary_mod, lifestyle_prompt, escalate}
Reward : Equation 1 — R_t^clinical + lambda_adh*R_t^adh + lambda_safe*R_t^safety
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.constraint_set import ConstraintSet
from envs.reward_function import compute_reward
from envs.spaces import ACTION_NAMES, N_ACTIONS, STATE_DIM
from knowledge.policy_registry import PolicyRegistry
from preprocessing.signal_pipeline import SignalPipeline


class PatientEnv(gym.Env):
    """Single-patient Gymnasium environment for AgHealth+."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        patient_data: dict,
        config: dict,
        policy_registry: PolicyRegistry,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.patient_data = patient_data
        self.config = config
        self.P = policy_registry
        self.constraint_set = ConstraintSet(policy_registry)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._vitals_list = patient_data.get("vitals", [])
        self.max_steps = len(self._vitals_list)
        self.current_step = 0
        self.episode_reward = 0.0
        self._action_mask = np.ones(N_ACTIONS, dtype=bool)

        # Lazy-import to avoid circular deps
        try:
            self._pipeline = SignalPipeline()
        except Exception:
            self._pipeline = None

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        if self.current_step >= len(self._vitals_list):
            return np.zeros(STATE_DIM, dtype=np.float32)
        idx = self.current_step
        vit = self._vitals_list[idx]

        # Raw vitals — crude z-score
        vitals_z = np.array(
            [
                (vit.get("sbp", 130) - 130) / 20,
                (vit.get("dbp", 82) - 82) / 12,
                (vit.get("glucose_mgdl", 110) - 110) / 40,
                (vit.get("heart_rate", 72) - 72) / 12,
                (vit.get("spo2", 97.5) - 97.5) / 2,
            ],
            dtype=np.float32,
        )

        # Rolling means (5-step window)
        win_start = max(0, idx - 5)
        window_vits = self._vitals_list[win_start : idx + 1]
        rolling_mean = np.mean(
            [
                [
                    w.get("sbp", 130),
                    w.get("dbp", 82),
                    w.get("glucose_mgdl", 110),
                    w.get("heart_rate", 72),
                    w.get("spo2", 97.5),
                ]
                for w in window_vits
            ],
            axis=0,
        ).astype(np.float32) / np.array([130, 82, 110, 72, 97.5])

        rolling_slope = vitals_z - (rolling_mean - 1.0)

        # Adherence (3)
        adherence = np.array(
            [
                vit.get("adherence_med", 0.7),
                vit.get("adherence_diet", 0.5),
                vit.get("adherence_lifestyle", 0.6),
            ],
            dtype=np.float32,
        )

        # Context (4)
        t_min = vit.get("t_minutes", idx * 5)
        t_h = (t_min / 60) % 24
        context = np.array(
            [
                t_h / 24,
                (t_h // 24 % 7) / 7,
                float(abs(t_h % 4 - 2) < 0.5),  # near meal
                vit.get("adherence_lifestyle", 0.6),
            ],
            dtype=np.float32,
        )

        # Policy flags (3)
        conditions = self.patient_data.get("demographics", {})
        policies_arr = np.array(
            [
                float(conditions.get("hypertension", False)),
                float(
                    self.patient_data.get("policies", {}).get(
                        "caffeine_restriction", False
                    )
                ),
                float(conditions.get("ckd", False)),
            ],
            dtype=np.float32,
        )

        return np.clip(
            np.concatenate(
                [
                    vitals_z,
                    rolling_mean - 1.0,
                    rolling_slope,
                    adherence,
                    context,
                    policies_arr,
                ]
            ),
            -10.0,
            10.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Action masking (for MaskablePPO)
    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        if self.current_step >= len(self._vitals_list):
            return np.ones(N_ACTIONS, dtype=bool)
        vit = self._vitals_list[self.current_step]
        adh = vit.get("adherence_med", 0.7)
        mask = np.ones(N_ACTIONS, dtype=bool)
        if not (self.P.should_escalate(vit) or self.P.should_watch(vit)):
            mask[4] = False
        if adh > 0.92:
            mask[1] = False
        self._action_mask = mask
        return mask

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def step(self, action: int):
        mask = self.action_masks()
        if not mask[action]:
            # Constraint violation — redirect to no_action
            action = 0

        vit = (
            self._vitals_list[self.current_step]
            if self.current_step < len(self._vitals_list)
            else {}
        )

        reward = compute_reward(
            state=self._get_obs(),
            action=action,
            vital=vit,
            config=self.config.get(
                "reward",
                {
                    "lambda_adherence": 0.3,
                    "lambda_safety": 2.0,
                    "clinical_stability_weight": 1.0,
                    "bp_target_systolic": [120, 130],
                    "glucose_tir_target": [70, 180],
                    "constraint_violation_penalty": -10.0,
                    "escalation_event_penalty": -5.0,
                },
            ),
            action_mask=mask,
        )
        self.episode_reward += reward
        self.current_step += 1
        done = self.current_step >= self.max_steps

        obs = self._get_obs() if not done else np.zeros(STATE_DIM, dtype=np.float32)
        info = {
            "action_name": ACTION_NAMES.get(action, "unknown"),
            "sbp": vit.get("sbp", 0),
            "glucose": vit.get("glucose_mgdl", 0),
            "episode_reward": self.episode_reward,
            "is_emergency": bool(vit.get("event_type")),
        }
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        self._action_mask = np.ones(N_ACTIONS, dtype=bool)
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.current_step < len(self._vitals_list):
                vit = self._vitals_list[self.current_step]
                print(
                    f"Step {self.current_step:5d} | "
                    f"SBP={vit.get('sbp', 0):5.1f} | "
                    f"Glu={vit.get('glucose_mgdl', 0):5.1f} | "
                    f"HR={vit.get('heart_rate', 0):4.1f} | "
                    f"SpO2={vit.get('spo2', 0):4.1f}%"
                )
