from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real" / "processed"
OUT = ROOT / "evaluation" / "results" / "real"
FIG = ROOT / "evaluation" / "figures" / "real"
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
RED = "#d62728"
GREEN = "#2ca02c"
PURPLE = "#9467bd"
BLACK = "#111111"
PALETTE = [BLUE, ORANGE, RED, GREEN, PURPLE]


def frame(ax):
    for item in ax.spines.values():
        item.set_visible(True)
        item.set_color(BLACK)
        item.set_linewidth(1.4)
    ax.tick_params(width=1.2, color=BLACK)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)


def main():
    train = pd.read_csv(DATA / "ppg_dalia_train_split.csv")
    x_train = train[FEATURES]
    y_train = train["high_risk"].astype(int)
    model = Pipeline([("scale", StandardScaler()), ("model", GaussianNB())])
    model.fit(x_train, y_train)
    raw = model.named_steps["model"]
    diff = np.abs(raw.theta_[1] - raw.theta_[0])
    scale = np.sqrt(np.maximum(raw.var_[0] + raw.var_[1], 1e-12))
    imp = pd.DataFrame({"feature": FEATURES, "value": diff / scale}).sort_values("value", ascending=True)
    imp.to_csv(OUT / "ppg_dalia_feature_importance.csv", index=False)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(imp))]
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.barh(imp["feature"], imp["value"], color=colors, edgecolor=BLACK, linewidth=1.1)
    ax.set_title("PPG-DaLiA Feature Importance")
    ax.set_xlabel("Importance")
    frame(ax)
    plt.tight_layout()
    plt.savefig(FIG / "ppg_dalia_feature_importance.png", dpi=600, bbox_inches="tight")
    plt.savefig(FIG / "ppg_dalia_feature_importance.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
