"""
generate_patients.py
====================
Generates the synthetic longitudinal patient cohort (Section 3.6 of paper).
Calibrated to NHANES 2017-2020 and WHO epidemiological distributions.

Usage:
    python data/synthetic/generate_patients.py
    python data/synthetic/generate_patients.py --sample 10   # quick test
    python data/synthetic/generate_patients.py --config configs/patient_sim.yaml

Output (data/synthetic/cohort/):
    patients_static.csv       — demographics + conditions (one row per patient)
    vitals_longitudinal.csv   — timestamped signals (one row per reading)
    medications.csv           — medication schedules
    events.csv                — rare event ground-truth labels
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from tqdm import tqdm
except ImportError:
    # fallback if tqdm is not installed
    def tqdm(x, **kwargs):
        return x


# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.synthetic.adherence_model import AdherenceModel
from data.synthetic.hazard_model import HazardModel


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def sample_demographics(cfg: dict, rng: np.random.Generator) -> dict:
    """Sample one patient's demographics from calibrated distributions."""
    age = float(
        np.clip(
            rng.normal(
                cfg["demographics"]["age"]["mean"], cfg["demographics"]["age"]["std"]
            ),
            cfg["demographics"]["age"]["min"],
            cfg["demographics"]["age"]["max"],
        )
    )
    sex = "F" if rng.random() < cfg["demographics"]["sex_ratio_female"] else "M"
    bmi = float(
        np.clip(
            rng.normal(
                cfg["demographics"]["bmi"]["mean"], cfg["demographics"]["bmi"]["std"]
            ),
            cfg["demographics"]["bmi"]["min"],
            cfg["demographics"]["bmi"]["max"],
        )
    )
    conditions = {
        cond: bool(rng.random() < prob) for cond, prob in cfg["conditions"].items()
    }
    conditions["obesity"] = bmi >= 30.0
    return {"age": round(age, 1), "sex": sex, "bmi": round(bmi, 1), **conditions}


def sample_baseline_vitals(patient: dict, cfg: dict, rng: np.random.Generator) -> dict:
    """Sample baseline vitals conditioned on clinical conditions."""
    v = cfg["vitals"]
    hyp = patient.get("hypertension", False)
    dm = patient.get("type2_diabetes", False)
    sbp_m = v["systolic_bp"]["mean_by_condition"]["hypertension" if hyp else "healthy"]
    dbp_m = v["diastolic_bp"]["mean_by_condition"]["hypertension" if hyp else "healthy"]
    glc_m = v["fasting_glucose_mgdl"]["mean_by_condition"][
        "diabetes" if dm else "healthy"
    ]
    return {
        "sbp_baseline": float(
            np.clip(
                rng.normal(sbp_m, v["systolic_bp"]["std"]),
                v["systolic_bp"]["min"],
                v["systolic_bp"]["max"],
            )
        ),
        "dbp_baseline": float(
            np.clip(
                rng.normal(dbp_m, v["diastolic_bp"]["std"]),
                v["diastolic_bp"]["min"],
                v["diastolic_bp"]["max"],
            )
        ),
        "glucose_baseline": float(
            np.clip(
                rng.normal(glc_m, v["fasting_glucose_mgdl"]["std"]),
                v["fasting_glucose_mgdl"]["min"],
                v["fasting_glucose_mgdl"]["max"],
            )
        ),
        "hr_baseline": float(
            np.clip(
                rng.normal(v["heart_rate_bpm"]["mean"], v["heart_rate_bpm"]["std"]),
                v["heart_rate_bpm"]["min"],
                v["heart_rate_bpm"]["max"],
            )
        ),
        "spo2_baseline": float(
            np.clip(
                rng.normal(v["spo2_percent"]["mean"], v["spo2_percent"]["std"]),
                v["spo2_percent"]["min"],
                v["spo2_percent"]["max"],
            )
        ),
    }


def assign_medications(patient: dict, rng: np.random.Generator) -> list:
    """Assign evidence-based medication regimens from clinical conditions."""
    meds = []
    if patient.get("hypertension"):
        meds.append(
            {
                "drug": "ACE_inhibitor",
                "dose_mg": int(rng.choice([5, 10, 20])),
                "frequency": "once_daily",
                "timing": "morning",
                "renal_adjustment": bool(patient.get("ckd", False)),
                "sodium_cap_gday": 2.3,
            }
        )
    if patient.get("type2_diabetes"):
        meds.append(
            {
                "drug": "metformin",
                "dose_mg": int(rng.choice([500, 1000])),
                "frequency": "twice_daily",
                "timing": "with_meals",
                "contraindicated_egfr_below": 30,
            }
        )
    if patient.get("hyperlipidemia"):
        meds.append(
            {
                "drug": "statin",
                "dose_mg": int(rng.choice([20, 40])),
                "frequency": "once_daily",
                "timing": "evening",
                "hepatic_monitoring": True,
            }
        )
    if patient.get("hypertension") and (
        patient.get("type2_diabetes") or patient.get("ckd")
    ):
        meds.append(
            {
                "drug": "ARB",
                "dose_mg": int(rng.choice([50, 100])),
                "frequency": "once_daily",
                "timing": "morning",
            }
        )
    return meds


def generate_longitudinal_vitals(
    pid: int,
    patient: dict,
    baseline: dict,
    cfg: dict,
    adherence_model: AdherenceModel,
    hazard_model: HazardModel,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate timestamped vital signs using autoregressive model with
    circadian rhythms, medication effects, and rare events."""
    timestep = cfg["timestep_minutes"]
    total_min = cfg["simulation_months"] * 30 * 24 * 60
    n_steps = total_min // timestep

    records = []
    sbp = baseline["sbp_baseline"]
    dbp = baseline["dbp_baseline"]
    glc = baseline["glucose_baseline"]
    hr = baseline["hr_baseline"]
    spo2 = baseline["spo2_baseline"]

    for step in range(n_steps):
        t_min = step * timestep
        t_hours = t_min / 60
        t_days = t_hours / 24

        adh = adherence_model.get_adherence(pid, t_days)

        # Age-related slow drift
        age_drift = 1.0 + (patient["age"] / 100.0) * (t_days / 365.0) * 0.015

        # Circadian (simplified sinusoidal)
        circ_bp = 8.0 * np.sin(2 * np.pi * (t_hours % 24 - 6) / 24)
        circ_hr = 12.0 * np.sin(2 * np.pi * (t_hours % 24 - 8) / 24)

        # Medication effects when adherent
        med_bp = -15.0 * adh["medication"] if patient.get("hypertension") else 0.0
        med_glc = -25.0 * adh["medication"] if patient.get("type2_diabetes") else 0.0

        # Autoregressive update + sensor noise
        sbp = float(
            np.clip(
                0.98 * sbp
                + 0.02 * (baseline["sbp_baseline"] * age_drift + med_bp + circ_bp)
                + rng.normal(0, 3),
                80,
                260,
            )
        )
        dbp = float(np.clip(sbp * 0.62 + rng.normal(0, 2), 40, 150))
        glc = float(
            np.clip(
                0.97 * glc
                + 0.03 * (baseline["glucose_baseline"] + med_glc)
                + rng.normal(0, 5),
                40,
                500,
            )
        )
        hr = float(
            np.clip(
                0.95 * hr
                + 0.05 * (baseline["hr_baseline"] + circ_hr)
                + rng.normal(0, 2),
                40,
                160,
            )
        )
        spo2 = float(np.clip(0.99 * spo2 + rng.normal(0, 0.3), 85, 100))

        event = hazard_model.check_event(pid, t_days, sbp, glc)

        records.append(
            {
                "patient_id": pid,
                "t_minutes": t_min,
                "sbp": round(sbp, 1),
                "dbp": round(dbp, 1),
                "glucose_mgdl": round(glc, 1),
                "heart_rate": round(hr, 1),
                "spo2": round(spo2, 1),
                "adherence_med": round(adh["medication"], 3),
                "adherence_diet": round(adh["dietary"], 3),
                "adherence_lifestyle": round(adh["lifestyle"], 3),
                "event_type": event,
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient cohort for AgHealth+/PAAI"
    )
    parser.add_argument("--config", default="configs/patient_sim.yaml")
    parser.add_argument(
        "--sample", type=int, default=None, help="Generate N patients only (quick test)"
    )
    parser.add_argument("--output_dir", default="data/synthetic/cohort")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = args.sample if args.sample else cfg["n_patients"]
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AgHealth+ Synthetic Cohort Generator")
    print(f"{'='*60}")
    print(f"Patients : {n}")
    print(
        f"Duration : {cfg['simulation_months']} months @ {cfg['timestep_minutes']}-min resolution"
    )
    print(f"Seed     : {cfg['seed']}")
    print(f"Output   : {args.output_dir}/\n")

    adh_model = AdherenceModel(cfg["adherence"], rng)
    haz_model = HazardModel(cfg["rare_events"], rng)

    patients_static, all_vitals, all_meds = [], [], []

    for pid in tqdm(range(n), desc="Generating patients", unit="pt"):
        patient = sample_demographics(cfg, rng)
        patient["patient_id"] = pid
        baseline = sample_baseline_vitals(patient, cfg, rng)
        patient.update(baseline)
        meds = assign_medications(patient, rng)
        patient["n_medications"] = len(meds)

        vitals_df = generate_longitudinal_vitals(
            pid, patient, baseline, cfg, adh_model, haz_model, rng
        )

        all_vitals.append(vitals_df)
        patients_static.append(patient)
        for m in meds:
            m["patient_id"] = pid
            all_meds.append(m)

    # Save outputs
    static_df = pd.DataFrame(patients_static)
    static_df.to_csv(f"{args.output_dir}/patients_static.csv", index=False)

    vitals_combined = pd.concat(all_vitals, ignore_index=True)
    vitals_combined.to_csv(f"{args.output_dir}/vitals_longitudinal.csv", index=False)

    meds_df = pd.DataFrame(all_meds)
    meds_df.to_csv(f"{args.output_dir}/medications.csv", index=False)

    events_df = vitals_combined[vitals_combined["event_type"].notna()][
        ["patient_id", "t_minutes", "event_type", "sbp", "glucose_mgdl"]
    ]
    events_df.to_csv(f"{args.output_dir}/events.csv", index=False)

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"  Total patients          : {len(static_df)}")
    print(f"  Vital sign readings     : {len(vitals_combined):,}")
    print(f"  Hypertension prevalence : {static_df['hypertension'].mean():.1%}")
    print(f"  Diabetes prevalence     : {static_df['type2_diabetes'].mean():.1%}")
    print(
        f"  Mean age                : {static_df['age'].mean():.1f} ± {static_df['age'].std():.1f} yr"
    )
    print(
        f"  Mean BMI                : {static_df['bmi'].mean():.1f} ± {static_df['bmi'].std():.1f}"
    )
    print(f"  Total rare events       : {len(events_df)}")
    if len(events_df):
        print(
            f"  Event breakdown         : {events_df['event_type'].value_counts().to_dict()}"
        )
    print(f"\n  Files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
