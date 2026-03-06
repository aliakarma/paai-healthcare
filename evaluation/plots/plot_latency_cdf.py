"""Figure 5 — CDF of alert latency."""

import numpy as np
import matplotlib.pyplot as plt
from evaluation.metrics import compute_latency_cdf


def plot_latency_cdf(results: dict, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {
        "aghealth": "#1a7ab5",
        "predictive_only": "#e87722",
        "rules_only": "#5ba85f",
    }
    labels = {
        "aghealth": "AgHealth+",
        "predictive_only": "Predictive-only",
        "rules_only": "Rules-only",
    }
    for method, label in labels.items():
        if method not in results:
            continue
        lat = results[method].get("latency", np.array([]))
        if len(lat) == 0:
            continue
        x, y = compute_latency_cdf(lat)
        ax.plot(x, y, color=colors[method], lw=2, label=label)

    ax.set_xlabel("Latency (s)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(
        "Alert Latency CDF (lower/faster is better)", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
