"""
plot_roc.py
===========
Figure 3 — ROC curves for anomaly / escalation detection.

Key corrections vs. previous version
--------------------------------------
* ROC curves are constructed from the actual ``y_true`` and ``y_score``
  arrays that live in the *results* dict returned by each evaluator.
  The previous version fabricated a synthetic ``y_true`` array by
  back-solving from a target AUC, producing curves that bore no
  relationship to the real model output.
* AUC values displayed in the legend are computed with
  ``sklearn.metrics.roc_auc_score`` from the true arrays, not hard-coded.
* The function validates that ``roc_scores`` and a shared ``y_true`` array
  exist in the results dict before attempting to plot, and emits a clear
  warning for any method whose data is absent.
* A shared ``y_true`` array is reconstructed from the event-set ground truth
  when it is not stored explicitly, using the same logic as the evaluators.
* ``save_path`` can be either a PDF or PNG path; the extension is respected.

Expected ``results`` dict structure (produced by run_evaluation.py)
--------------------------------------------------------------------
results = {
    "rules_only":      {"roc_scores": np.ndarray, "roc_auc": np.ndarray, ...},
    "predictive_only": {"roc_scores": np.ndarray, "roc_auc": np.ndarray, ...},
    "human_schedule":  {"roc_scores": np.ndarray, "roc_auc": np.ndarray, ...},
    "aghealth":        {"roc_scores": np.ndarray, "roc_auc": np.ndarray, ...},
    "_y_true":         np.ndarray,   # shared binary ground-truth (optional)
}

If ``_y_true`` is absent, ``cohort_dir`` must be supplied so the function
can reconstruct ground truth from the events CSV.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc as sklearn_auc


# ── Aesthetics ─────────────────────────────────────────────────────────────────
_COLORS = {
    "aghealth":        "#1a7ab5",
    "predictive_only": "#e87722",
    "rules_only":      "#5ba85f",
    "human_schedule":  "#999999",
}
_LABELS = {
    "aghealth":        "AgHealth+",
    "predictive_only": "Predictive-only (B2)",
    "rules_only":      "Rules-only (B1)",
    "human_schedule":  "Human-schedule (B3)",
}
_LINE_STYLES = {
    "aghealth":        ("-",  2.5),
    "predictive_only": ("--", 1.8),
    "rules_only":      ("--", 1.8),
    "human_schedule":  (":",  1.5),
}
_PLOT_ORDER = ["aghealth", "predictive_only", "rules_only", "human_schedule"]


def _load_y_true(cohort_dir: str) -> np.ndarray:
    """Reconstruct the per-timestep binary ground-truth from cohort CSVs.

    This mirrors the logic in every ``evaluate()`` function so that the
    resulting ``y_true`` array has the same ordering as ``y_score``.
    """
    import pandas as pd

    vitals_path = Path(cohort_dir) / "vitals_longitudinal.csv"
    events_path = Path(cohort_dir) / "events.csv"

    if not vitals_path.exists() or not events_path.exists():
        raise FileNotFoundError(
            f"Cohort files not found in '{cohort_dir}'. "
            "Run: python data/synthetic/generate_patients.py"
        )

    vitals_df = pd.read_csv(vitals_path)
    events_df  = pd.read_csv(events_path)

    event_set = set(zip(
        events_df["patient_id"].astype(int),
        events_df["t_minutes"].astype(int),
    ))

    y_true = np.array([
        1 if (int(r["patient_id"]), int(r["t_minutes"])) in event_set else 0
        for _, r in vitals_df.iterrows()
    ], dtype=int)

    return y_true


def plot_roc(
    results: dict,
    save_path: Optional[str] = None,
    cohort_dir: Optional[str] = None,
) -> None:
    """Plot ROC curves for all methods from their actual score arrays.

    Parameters
    ----------
    results    : dict mapping method key → metric dict (see module docstring).
    save_path  : optional file path for saving (PDF or PNG).
    cohort_dir : path to the synthetic cohort directory.  Required only when
                 ``results`` does not contain a ``"_y_true"`` key.
    """
    # ── Resolve shared ground-truth ───────────────────────────────────────────
    if "_y_true" in results and results["_y_true"] is not None:
        y_true_shared = np.asarray(results["_y_true"], dtype=int)
    elif cohort_dir is not None:
        y_true_shared = _load_y_true(cohort_dir)
    else:
        # Last resort: attempt to infer from results length parity.
        # This is only possible when all methods share the same y_true.
        y_true_shared = None

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))

    plotted_any = False

    for method in _PLOT_ORDER:
        if method not in results:
            warnings.warn(
                f"Method '{method}' not found in results dict — skipping.",
                UserWarning, stacklevel=2,
            )
            continue

        method_res = results[method]
        y_score = method_res.get("roc_scores")

        if y_score is None or len(y_score) == 0:
            warnings.warn(
                f"'{method}' has no 'roc_scores' — cannot plot ROC curve.",
                UserWarning, stacklevel=2,
            )
            continue

        y_score = np.asarray(y_score, dtype=float)

        # Determine y_true: prefer method-specific, then shared
        y_true_method = method_res.get("y_true")
        if y_true_method is not None:
            y_true = np.asarray(y_true_method, dtype=int)
        elif y_true_shared is not None:
            if len(y_true_shared) != len(y_score):
                warnings.warn(
                    f"Shared y_true length ({len(y_true_shared)}) != "
                    f"y_score length for '{method}' ({len(y_score)}). "
                    "Cannot plot this method.",
                    UserWarning, stacklevel=2,
                )
                continue
            y_true = y_true_shared
        else:
            warnings.warn(
                f"No y_true available for '{method}'. "
                "Pass cohort_dir= or include '_y_true' in results.",
                UserWarning, stacklevel=2,
            )
            continue

        if len(np.unique(y_true)) < 2:
            warnings.warn(
                f"y_true for '{method}' has only one class — skipping.",
                UserWarning, stacklevel=2,
            )
            continue

        # ── Compute ROC curve and AUC from actual data ─────────────────────
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val      = roc_auc_score(y_true, y_score)

        # Prefer bootstrap AUC mean for the legend when available, as it
        # matches the value reported in Table 2.
        boot_aucs = method_res.get("roc_auc")
        if boot_aucs is not None and len(boot_aucs) > 0:
            auc_display   = float(np.mean(boot_aucs))
            auc_lo        = float(np.percentile(boot_aucs, 2.5))
            auc_hi        = float(np.percentile(boot_aucs, 97.5))
            legend_label  = (
                f"{_LABELS.get(method, method)}  "
                f"AUC={auc_display:.2f} [{auc_lo:.2f}, {auc_hi:.2f}]"
            )
        else:
            legend_label = (
                f"{_LABELS.get(method, method)}  AUC={auc_val:.2f}"
            )

        ls, lw = _LINE_STYLES.get(method, ("-", 1.5))
        ax.plot(
            fpr, tpr,
            color = _COLORS.get(method, "#333333"),
            lw    = lw,
            ls    = ls,
            label = legend_label,
        )
        plotted_any = True

    if not plotted_any:
        warnings.warn(
            "No methods could be plotted.  Check that results contain "
            "'roc_scores' arrays and that y_true is available.",
            UserWarning, stacklevel=2,
        )

    # ── Diagonal reference line ───────────────────────────────────────────────
    ax.plot([0, 1], [0, 1], color="black", lw=1., ls=":", alpha=0.5,
            label="Random classifier (AUC=0.50)")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — Escalation Detection", fontsize=12,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved ROC figure → {save_path}")

    plt.close()
