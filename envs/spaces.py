"""State and action space definitions for the patient Gym environment."""

import gymnasium as gym
import numpy as np

# 25-dimensional state: 5 z-scored vitals + 5 rolling means + 5 slopes
# + 3 adherence + 4 context + 3 policy flags
STATE_DIM = 25
N_ACTIONS = 5  # {no_action, med, diet, lifestyle, escalate}

STATE_SPACE = gym.spaces.Box(low=-10.0, high=10.0, shape=(STATE_DIM,), dtype=np.float32)
ACTION_SPACE = gym.spaces.Discrete(N_ACTIONS)
ACTION_NAMES = {
    0: "no_action",
    1: "medication_schedule",
    2: "dietary_modification",
    3: "lifestyle_prompt",
    4: "escalate",
}
