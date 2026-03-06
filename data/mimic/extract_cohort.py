"""
extract_cohort.py
=================
Extracts the hypertension+diabetes cohort from MIMIC-IV.

PREREQUISITES:
  1. Complete CITI training at citiprogram.org (2 hours, free)
  2. Apply for MIMIC-IV access at physionet.org/content/mimiciv/
  3. Download MIMIC-IV files to data/mimic/raw/  (NOT committed to git)
  4. pip install pandas pyarrow psycopg2-binary

IMPORTANT: Raw MIMIC data must NEVER be committed to this repository.
Only non-identifiable cohort summary stats are saved in data/mimic/extracted/.

Usage:
    python data/mimic/extract_cohort.py --config configs/mimic_extraction.yaml
"""

import argparse
import json
import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def check_mimic_available(mimic_dir: str, cfg: dict) -> bool:
    required = [
        f"{mimic_dir}/hosp/diagnoses_icd.csv.gz",
        f"{mimic_dir}/icu/icustays.csv.gz",
        f"{mimic_dir}/icu/chartevents.csv.gz",
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print("\nMIMIC-IV files not found:")
        for f in missing:
            print(f"  Missing: {f}")
        print("\nTo obtain MIMIC-IV access:")
        print("  1. Complete CITI training: https://citiprogram.org")
        print("  2. Apply at: https://physionet.org/content/mimiciv/")
        print("  3. Download to: data/mimic/raw/")
        print("  See: docs/mimic_setup.md for full step-by-step guide")
        return False
    return True


def load_target_patients(mimic_dir: str, cfg: dict) -> pd.DataFrame:
    codes = (
        cfg["inclusion_criteria"]["icd10_codes"]["hypertension"]
        + cfg["inclusion_criteria"]["icd10_codes"]["diabetes_t2"]
    )
    diag = pd.read_csv(
        f"{mimic_dir}/hosp/diagnoses_icd.csv.gz",
        usecols=["subject_id", "icd_code", "icd_version"],
    )
    diag = diag[diag["icd_version"] == 10]
    matched = diag[
        diag["icd_code"].apply(lambda c: any(str(c).startswith(code) for code in codes))
    ]["subject_id"].unique()
    print(f"  Patients with target ICD codes: {len(matched):,}")
    return matched


def load_icu_stays(mimic_dir: str, subject_ids, min_hours: int) -> pd.DataFrame:
    stays = pd.read_csv(
        f"{mimic_dir}/icu/icustays.csv.gz", parse_dates=["intime", "outtime"]
    )
    stays = stays[stays["subject_id"].isin(subject_ids)]
    stays["los_hours"] = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600
    return stays[stays["los_hours"] >= min_hours].reset_index(drop=True)


def extract_vitals(mimic_dir: str, stay_ids: list, itemids: dict) -> pd.DataFrame:
    all_ids = [i for ids in itemids.values() for i in ids]
    # Read in chunks for large file
    chunks = pd.read_csv(
        f"{mimic_dir}/icu/chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=500_000,
    )

    id_to_name = {i: k for k, ids in itemids.items() for i in ids}
    frames = []
    for chunk in chunks:
        sub = chunk[chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(all_ids)]
        if len(sub):
            sub = sub.copy()
            sub["vital_name"] = sub["itemid"].map(id_to_name)
            frames.append(sub)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Pivot and resample to 5-min grid
    result_frames = []
    for sid, grp in df.groupby("stay_id"):
        try:
            pivot = grp.pivot_table(
                index="charttime",
                columns="vital_name",
                values="valuenum",
                aggfunc="mean",
            )
            pivot = pivot.resample("5T").mean().interpolate(method="linear", limit=6)
            pivot["stay_id"] = sid
            result_frames.append(pivot)
        except Exception:
            continue
    return pd.concat(result_frames).reset_index() if result_frames else pd.DataFrame()


def load_ground_truth(mimic_dir: str, subject_ids) -> pd.DataFrame:
    events = []
    # Hypertensive urgency
    diag = pd.read_csv(f"{mimic_dir}/hosp/diagnoses_icd.csv.gz")
    hyp_codes = ["I16", "I16.0", "I16.1"]
    hyp_pts = diag[
        diag["icd_code"].apply(lambda c: any(str(c).startswith(x) for x in hyp_codes))
        & diag["subject_id"].isin(subject_ids)
    ]["subject_id"].unique()
    for sid in hyp_pts:
        events.append({"subject_id": int(sid), "event_type": "hypertensive_urgency"})
    # Hypoglycemia
    hypo_codes = ["E11.641", "E11.649"]
    hypo_pts = diag[
        diag["icd_code"].isin(hypo_codes) & diag["subject_id"].isin(subject_ids)
    ]["subject_id"].unique()
    for sid in hypo_pts:
        events.append({"subject_id": int(sid), "event_type": "hypoglycemia"})
    return pd.DataFrame(events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mimic_extraction.yaml")
    parser.add_argument("--mimic_dir", default="data/mimic/raw")
    parser.add_argument("--output", default="data/mimic/extracted")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if not check_mimic_available(args.mimic_dir, cfg):
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    print("\n" + "=" * 60)
    print("MIMIC-IV Cohort Extraction")
    print("=" * 60)

    print("\nStep 1: Loading diagnosis codes…")
    subject_ids = load_target_patients(args.mimic_dir, cfg)

    print("Step 2: Filtering ICU stays…")
    stays = load_icu_stays(
        args.mimic_dir, subject_ids, cfg["inclusion_criteria"]["icu_stay_hours_min"]
    )
    stays = stays.head(cfg["output"]["max_patients"])
    print(f"  Retained: {len(stays):,} stays")

    print("Step 3: Extracting vital signs (may take several minutes)…")
    vitals = extract_vitals(args.mimic_dir, stays["stay_id"].tolist(), cfg["itemids"])
    vitals.to_csv(f"{args.output}/vitals_mimic.csv", index=False)
    print(f"  Saved {len(vitals):,} readings → {args.output}/vitals_mimic.csv")

    print("Step 4: Loading ground truth events…")
    events = load_ground_truth(args.mimic_dir, stays["subject_id"].tolist())
    events.to_csv(f"{args.output}/events_mimic.csv", index=False)

    summary = {
        "n_stays": len(stays),
        "n_vital_readings": len(vitals),
        "n_escalation_events": len(events),
        "event_breakdown": (
            events["event_type"].value_counts().to_dict() if len(events) else {}
        ),
    }
    with open(f"{args.output}/cohort_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Extraction complete: {summary}")
    print("\nIMPORTANT: DO NOT commit files under data/mimic/raw/ or")
    print("data/mimic/extracted/ to git. See .gitignore.")


if __name__ == "__main__":
    main()
