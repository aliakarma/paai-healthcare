# RL Training Guide

## Quick Start
```bash
# 1. Generate synthetic cohort (if not done)
python data/synthetic/generate_patients.py

# 2. Start training (CPU, ~3-6 hours for 2M steps)
python rl/train.py

# 3. Monitor in real time
tensorboard --logdir rl/tensorboard/
# Open browser: http://localhost:6006

# 4. Evaluate trained policy
python evaluation/run_evaluation.py --mode synthetic
```

## Smoke Test (5 minutes)
```bash
python rl/train.py --sample 10   # 10 patients, 50k steps
```

## Resuming Training
```bash
python rl/train.py --resume rl/checkpoints/aghealth_ppo_1000000_steps.zip
```

## Key Hyperparameters (`configs/rl_training.yaml`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| total_timesteps | 2,000,000 | Total environment steps |
| n_envs | 8 | Parallel patient environments |
| gamma | 0.99 | Discount (long-term health) |
| lambda_safety | 2.0 | Safety reward weight |
| constraint_threshold | 0.05 | Max 5% violation rate |

## Convergence Criteria
- Episode reward should reach > 1.0 by 1M steps
- Constraint violation rate should drop below 5% by 500k steps
- If reward is stagnating, increase `ent_coef` to encourage exploration

## Action Masking
The environment uses `MaskablePPO` (sb3-contrib) which masks infeasible
actions at every timestep. This is the primary mechanism for enforcing
the CMDP constraint set C. Standard PPO will also work but may learn
to avoid masked actions more slowly.
