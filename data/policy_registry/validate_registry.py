"""
validate_registry.py
====================
Checks all policy registry files for internal consistency.
Run before deploying policy updates.

Usage:
    python data/policy_registry/validate_registry.py
"""
import json
import sys
from pathlib import Path

REGISTRY_DIR = Path(__file__).parent
REQUIRED_FILES = [
    "prescriber_rules.json",
    "allergy_exclusions.json",
    "escalation_criteria.json",
]

def validate_prescriber_rules(data: dict) -> list:
    errors = []
    if "sodium_cap_g_per_day" not in data:
        errors.append("Missing sodium_cap_g_per_day")
    if "timing_windows" not in data:
        errors.append("Missing timing_windows")
    for drug, cfg in data.get("dose_ceilings", {}).items():
        if not isinstance(cfg, (int, float)) or cfg <= 0:
            errors.append(f"Invalid dose ceiling for {drug}: {cfg}")
    return errors

def validate_escalation(data: dict) -> list:
    errors = []
    auto = data.get("automatic_escalation", {})
    watch = data.get("watch_and_repeat", {})
    sbp_auto = auto.get("systolic_bp_mmhg_gte", 999)
    sbp_watch = watch.get("systolic_bp_mmhg_gte", 999)
    if sbp_watch >= sbp_auto:
        errors.append(f"Watch threshold ({sbp_watch}) >= escalation threshold ({sbp_auto})")
    if not data.get("consent_required"):
        errors.append("consent_required should be true for HIPAA/GDPR compliance")
    return errors

def main():
    all_errors = []
    for fname in REQUIRED_FILES:
        fpath = REGISTRY_DIR / fname
        if not fpath.exists():
            all_errors.append(f"MISSING FILE: {fname}")
            continue
        with open(fpath) as f:
            data = json.load(f)
        if fname == "prescriber_rules.json":
            errors = validate_prescriber_rules(data)
        elif fname == "escalation_criteria.json":
            errors = validate_escalation(data)
        else:
            errors = []
        for e in errors:
            all_errors.append(f"{fname}: {e}")

    if all_errors:
        print("VALIDATION FAILED:")
        for e in all_errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"✓ All {len(REQUIRED_FILES)} registry files passed validation.")

if __name__ == "__main__":
    main()
