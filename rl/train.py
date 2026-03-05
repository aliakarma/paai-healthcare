"""
train.py
========
Main training script for the AgHealth+ constrained RL policy.

Usage:
    python rl/train.py
    python rl/train.py --config configs/rl_training.yaml
    python rl/train.py --sample 10   # quick smoke test on 10 patients
    python rl/train.py --resume rl/checkpoints/aghealth_ppo_500000_steps.zip

Hardware: CPU sufficient — 2M steps on 8 parallel envs takes ~3-6 hours.
Monitor training: tensorboard --logdir rl/tensorboard/
"""
import argparse
import os
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_patient_data(cohort_dir: str) -> list:
    import pandas as pd
    static  = pd.read_csv(f"{cohort_dir}/patients_static.csv")
    vitals  = pd.read_csv(f"{cohort_dir}/vitals_longitudinal.csv")
    patients = []
    for _, row in static.iterrows():
        pid = int(row["patient_id"])
        pv  = vitals[vitals["patient_id"] == pid].to_dict("records")
        patients.append({
            "patient_id": pid,
            "demographics": row.to_dict(),
            "vitals": pv,
            "policies": {
                "sodium_cap":          bool(row.get("hypertension", False)),
                "caffeine_restriction": True,
                "renal_adjustment":    bool(row.get("ckd", False)),
            },
        })
    return patients


def make_env(patient_data_list, config, policy_registry, rank, seed=42):
    from stable_baselines3.common.utils import set_random_seed
    def _init():
        from envs.patient_env import PatientEnv
        pt = patient_data_list[rank % len(patient_data_list)]
        env = PatientEnv(pt, config, policy_registry)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train AgHealth+ RL policy")
    parser.add_argument("--config",         default="configs/rl_training.yaml")
    parser.add_argument("--patient_config", default="configs/patient_sim.yaml")
    parser.add_argument("--cohort_dir",     default="data/synthetic/cohort")
    parser.add_argument("--resume",         default=None)
    parser.add_argument("--sample",         type=int, default=None,
                        help="Limit to N patients (quick test)")
    parser.add_argument("--device",         default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        rl_cfg = yaml.safe_load(f)
    with open(args.patient_config) as f:
        pt_cfg = yaml.safe_load(f)
    with open("configs/preprocessing.yaml") as f:
        pre_cfg = yaml.safe_load(f)
    with open("configs/escalation_thresholds.yaml") as f:
        esc_cfg = yaml.safe_load(f)

    merged = {**pt_cfg, **rl_cfg, "preprocessing": pre_cfg,
               "escalation_thresholds": esc_cfg}

    # Ensure cohort exists
    if not Path(f"{args.cohort_dir}/patients_static.csv").exists():
        print(f"Cohort not found at {args.cohort_dir}.")
        print("Run first: python data/synthetic/generate_patients.py")
        sys.exit(1)

    print("Loading patient cohort…")
    patients = load_patient_data(args.cohort_dir)
    if args.sample:
        patients = patients[:args.sample]
        print(f"Quick-test mode: using {args.sample} patients")
    print(f"Loaded {len(patients)} patients")

    from knowledge.policy_registry import PolicyRegistry
    registry = PolicyRegistry()

    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
        from stable_baselines3.common.callbacks import (
            CheckpointCallback, EvalCallback)
        from rl.callbacks import TensorboardRewardCallback
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install stable-baselines3 sb3-contrib")
        sys.exit(1)

    n_envs = min(rl_cfg["n_envs"], len(patients))
    env_fns = [make_env(patients, merged, registry, i) for i in range(n_envs)]
    vec_env = VecMonitor(SubprocVecEnv(env_fns))

    os.makedirs(rl_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(rl_cfg["tensorboard_dir"], exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, rl_cfg["checkpoint_freq"] // n_envs),
        save_path=rl_cfg["checkpoint_dir"],
        name_prefix="aghealth_ppo", verbose=1)

    eval_env_fn = make_env(patients, merged, registry, 0)
    eval_env = eval_env_fn()
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=max(1, rl_cfg["eval_freq"] // n_envs),
        n_eval_episodes=rl_cfg["eval_episodes"],
        best_model_save_path=rl_cfg["checkpoint_dir"] + "best/",
        verbose=1)

    # Use MaskablePPO if available (strongly recommended)
    try:
        from sb3_contrib import MaskablePPO as PPOClass
        print("Using MaskablePPO ✓")
    except ImportError:
        from stable_baselines3 import PPO as PPOClass
        print("sb3-contrib not installed — using standard PPO")
        print("Install: pip install sb3-contrib")

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPOClass.load(args.resume, env=vec_env, device=args.device)
    else:
        model = PPOClass(
            "MlpPolicy", vec_env,
            learning_rate  = rl_cfg["learning_rate"],
            n_steps        = rl_cfg["n_steps"],
            batch_size     = rl_cfg["batch_size"],
            n_epochs       = rl_cfg["n_epochs"],
            gamma          = rl_cfg["gamma"],
            gae_lambda     = rl_cfg["gae_lambda"],
            clip_range     = rl_cfg["clip_range"],
            ent_coef       = rl_cfg["ent_coef"],
            vf_coef        = rl_cfg["vf_coef"],
            max_grad_norm  = rl_cfg["max_grad_norm"],
            tensorboard_log= rl_cfg["tensorboard_dir"],
            verbose=1, device=args.device)

    total = rl_cfg["total_timesteps"] if not args.sample else 50_000
    print(f"\nStarting training — {total:,} steps across {n_envs} envs")
    print(f"Monitor: tensorboard --logdir {rl_cfg['tensorboard_dir']}")
    model.learn(
        total_timesteps=total,
        callback=[checkpoint_cb, eval_cb, TensorboardRewardCallback()],
        tb_log_name="aghealth_ppo",
        reset_num_timesteps=(args.resume is None))

    final = os.path.join(rl_cfg["checkpoint_dir"], "aghealth_final")
    model.save(final)
    print(f"\nTraining complete. Final model: {final}.zip")


if __name__ == "__main__":
    main()
