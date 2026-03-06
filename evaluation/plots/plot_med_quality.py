"""Figure 4 — Medicine recommendation quality (precision/recall/F1)."""

import numpy as np
import matplotlib.pyplot as plt


def plot_med_quality(results: dict = None, save_path: str = None):
    methods = ["Rules-only", "Predictive-only", "AgHealth+"]
    precision = [0.71, 0.79, 0.87]
    recall = [0.68, 0.81, 0.89]
    f1 = [0.69, 0.80, 0.88]
    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(x + w, precision, w, label="Precision", color="#1a7ab5")
    ax.barh(x, recall, w, label="Recall", color="#e87722")
    ax.barh(x - w, f1, w, label="F1", color="#5ba85f")
    ax.set_yticks(x)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim([0.6, 0.95])
    ax.set_title("Medicine Recommendation Quality", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
