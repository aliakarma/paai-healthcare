"""
run_evaluation.py
=================
Master evaluation script — run this to reproduce ALL paper results.

Usage:
    python evaluation/run_evaluation.py --mode synthetic    # Table 2, Figures 3-7
    python evaluation/run_evaluation.py --mode mimic        # MIMIC-IV validation
    python evaluation/run_evaluation.py --mode all          # Everything
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


METHODS = ["rules_only", "predictive_only", "human_schedule", "aghealth"]
LABELS = {
    "rules_only": "Rules-only (B1)",
    "predictive_only": "Predictive-only (B2)",
    "human_schedule": "Human-schedule (B3)",
    "aghealth": "AgHealth+",
}


def run_synthetic(cohort_dir: str, model_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("SYNTHETIC COHORT EVALUATION")
    print("=" * 60)

    from baselines.human_schedule import evaluate as eval_b3
    from baselines.predictive_only import evaluate as eval_b2
    from baselines.rules_only import evaluate as eval_b1
    from evaluation.ablation import run_ablation
    from evaluation.plots.plot_adherence import plot_adherence
    from evaluation.plots.plot_latency_cdf import plot_latency_cdf
    from evaluation.plots.plot_learning_curves import plot_learning_curves
    from evaluation.plots.plot_med_quality import plot_med_quality
    from evaluation.plots.plot_roc import plot_roc
    from evaluation.statistical_tests import bonferroni_correct, delong_test
    from rl.evaluate_policy import evaluate_aghealth

    print("\nRunning baselines and AgHealth+…")
    results = {
        "rules_only": eval_b1(cohort_dir),
        "predictive_only": eval_b2(cohort_dir),
        "human_schedule": eval_b3(cohort_dir),
        "aghealth": evaluate_aghealth(cohort_dir, model_path),
    }

    # ── Table 2 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 2: PRIMARY OUTCOMES (mean ± sd, 95% CI)")
    print("=" * 60)
    rows = []
    for m in METHODS:
        r = results[m]
        auc_mean = float(np.mean(r["roc_auc"]))
        auc_lo = float(np.percentile(r["roc_auc"], 2.5))
        auc_hi = float(np.percentile(r["roc_auc"], 97.5))
        acc_m = float(np.mean(r["accuracy"]))
        acc_s = float(np.std(r["accuracy"]))
        lat_med = float(np.median(r["latency"]))
        prec_m = float(np.mean(r["med_precision"]))
        if m != "aghealth":
            pv = bonferroni_correct(
                delong_test(r["roc_auc"], results["aghealth"]["roc_auc"]), n_tests=3
            )
            pv_str = f"< {pv:.3f}"
        else:
            pv_str = "—"
        rows.append(
            {
                "Method": LABELS[m],
                "Accuracy": f"{acc_m:.2f} ± {acc_s:.2f}",
                "ROC AUC": f"{auc_mean:.2f} [{auc_lo:.2f}, {auc_hi:.2f}]",
                "Med. Latency (s)": f"{lat_med:.1f}",
                "Med. Precision": f"{prec_m:.2f}",
                "p vs AgHealth+": pv_str,
            }
        )
    table2 = pd.DataFrame(rows)
    print(table2.to_string(index=False))
    table2.to_csv(f"{output_dir}/table2_primary_outcomes.csv", index=False)

    # ── Table 3: Ablation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 3: ABLATION STUDY")
    print("=" * 60)
    ablation = run_ablation(cohort_dir, model_path)
    abl_df = pd.DataFrame(ablation)
    print(abl_df.to_string(index=False))
    abl_df.to_csv(f"{output_dir}/table3_ablation.csv", index=False)

    # ── Figures ───────────────────────────────────────────────
    print("\nGenerating figures…")
    plot_roc(results, f"{output_dir}/fig3_roc.pdf")
    plot_med_quality(results, f"{output_dir}/fig4_med_quality.pdf")
    plot_latency_cdf(results, f"{output_dir}/fig5_latency_cdf.pdf")
    plot_adherence(results, f"{output_dir}/fig6_adherence.pdf")
    plot_learning_curves(save_path=f"{output_dir}/fig7_learning_curves.pdf")

    print(f"\n✓ All results saved to {output_dir}/")


def run_mimic(mimic_dir: str, model_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("MIMIC-IV REAL DATA VALIDATION")
    print("=" * 60)
    from evaluation.mimic_evaluation import evaluate_mimic

    vp = f"{mimic_dir}/extracted/vitals_mimic.csv"
    ep = f"{mimic_dir}/extracted/events_mimic.csv"
    if not Path(vp).exists():
        print(f"MIMIC data not found: {vp}")
        print("Run: python data/mimic/extract_cohort.py")
        return
    res = evaluate_mimic(vp, ep)
    print(f"\nMIMIC-IV Anomaly Detection:")
    print(f"  Patients       : {res.get('n_patients')}")
    print(
        f"  ROC AUC        : {res.get('roc_auc', 0):.3f} "
        f"[{res['auc_ci'][0]:.3f}, {res['auc_ci'][1]:.3f}]"
    )
    print(f"  Sensitivity    : {res.get('sensitivity', 0):.3f}")
    print(f"  Specificity    : {res.get('specificity', 0):.3f}")
    print(f"  False positive : {res.get('fpr', 0):.3f}")
    with open(f"{output_dir}/mimic_validation.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n✓ MIMIC results → {output_dir}/mimic_validation.json")


def main():
    parser = argparse.ArgumentParser(description="AgHealth+ full evaluation")
    parser.add_argument(
        "--mode", choices=["synthetic", "mimic", "all"], default="synthetic"
    )
    parser.add_argument("--cohort_dir", default="data/synthetic/cohort")
    parser.add_argument("--mimic_dir", default="data/mimic")
    parser.add_argument("--model_path", default="rl/checkpoints/best/best_model.zip")
    parser.add_argument("--output_dir", default="evaluation/results")
    args = parser.parse_args()

    if not Path(f"{args.cohort_dir}/vitals_longitudinal.csv").exists():
        print(
            f"Cohort not found. Run first:\n"
            f"  python data/synthetic/generate_patients.py"
        )
        sys.exit(1)

    if args.mode in ("synthetic", "all"):
        run_synthetic(args.cohort_dir, args.model_path, args.output_dir)
    if args.mode in ("mimic", "all"):
        run_mimic(args.mimic_dir, args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
