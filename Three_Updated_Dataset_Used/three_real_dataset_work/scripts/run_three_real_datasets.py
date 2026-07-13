from pathlib import Path
import io
import json
import pickle
import shutil
import warnings
import zipfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore")


ROOT = Path(globals().get("__file__", Path.cwd())).resolve().parent
if "__file__" not in globals():
    ROOT = Path.cwd().resolve()
if ROOT.name in ("scripts", "notebooks"):
    ROOT = ROOT.parent
BASE = ROOT.parent
RAW = BASE / "data"
DATA_DIR = ROOT / "data" / "processed"
OUT = ROOT / "evaluation" / "results"
FIG = ROOT / "evaluation" / "figures"
NOTEBOOK_DIR = ROOT / "notebooks"
CONFIG_DIR = ROOT / "configs"
DOCS_DIR = ROOT / "docs"
SEED = 42
DPI = 600
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
RED = "#d62728"
GREEN = "#2ca02c"
PURPLE = "#9467bd"
BLACK = "#111111"
PALETTE = [BLUE, ORANGE, RED, GREEN, PURPLE]
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
METHODS = ["Rules-only (B1)", "Predictive-only (B2)", "Human-schedule (B3)", "AgHealth+"]


def make_dirs():
    for path in [DATA_DIR, OUT, FIG, NOTEBOOK_DIR, CONFIG_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_pickle_from_zip(zip_path, name):
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(name)
    return pickle.load(io.BytesIO(raw), encoding="latin1")


def clean_number(value):
    value = np.asarray(value, dtype=float)
    if value.size == 0:
        return 0.0
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).mean())


def std_number(value):
    value = np.asarray(value, dtype=float)
    if value.size == 0:
        return 0.0
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).std())


def slope_number(value):
    value = np.asarray(value, dtype=float).reshape(-1)
    if value.size < 2:
        return 0.0
    return float(np.nan_to_num(value[-1] - value[0], nan=0.0, posinf=0.0, neginf=0.0))


def mode_label(values):
    values = np.asarray(values).astype(int)
    values = values[values > 0]
    if values.size == 0:
        return 0
    ids, counts = np.unique(values, return_counts=True)
    return int(ids[np.argmax(counts)])


def patient_id(path):
    return path.name.split("-")[0]


def read_ohio_file(path, split_name):
    root = ET.parse(path).getroot()
    pid = patient_id(path)
    weight = float(root.attrib.get("weight", 0))
    events = []
    node = root.find("glucose_level")
    if node is None:
        return pd.DataFrame()
    for item in node.findall("event"):
        ts = pd.to_datetime(item.attrib["ts"], format="%d-%m-%Y %H:%M:%S")
        val = float(item.attrib["value"])
        events.append((ts, val))
    df = pd.DataFrame(events, columns=["time", "glucose"]).sort_values("time")
    rows = []
    values = df["glucose"].values
    times = df["time"].values
    for i in range(12, len(df) - 6, 3):
        past = values[i - 12 : i + 1]
        future = values[i + 1 : i + 7]
        high = np.maximum(future - 180.0, 0.0)
        low = np.maximum(70.0 - future, 0.0)
        score = float(np.max(high + low))
        hour = pd.Timestamp(times[i]).hour + pd.Timestamp(times[i]).minute / 60.0
        rows.append(
            {
                "dataset": "OhioT1DM",
                "subject": pid,
                "split": split_name,
                "age": hour / 24.0,
                "sex": weight,
                "bmi": values[i],
                "bp": clean_number(past),
                "s1": std_number(past),
                "s2": slope_number(past),
                "s3": float(np.min(past)),
                "s4": float(np.max(past)),
                "s5": float(np.max(past) - np.min(past)),
                "s6": float(i / len(df)),
                "progression_score": score,
                "high_risk": int(score > 0),
            }
        )
    return pd.DataFrame(rows)


def build_ohio():
    folder = RAW / "OhioT1DM"
    parts = []
    for path in sorted(folder.glob("*-training.xml")):
        parts.append(read_ohio_file(path, "train"))
    for path in sorted(folder.glob("*-testing.xml")):
        parts.append(read_ohio_file(path, "test"))
    df = pd.concat(parts, ignore_index=True)
    return df, "future 30 minute hypo/hyperglycemia event", 0.0, "official training/testing XML split"


def wesad_members():
    with zipfile.ZipFile(RAW / "WESAD.zip") as z:
        names = z.namelist()
    return [name for name in names if name.endswith(".pkl") and "/S" in name]


def build_wesad_subject(zip_path, name):
    data = read_pickle_from_zip(zip_path, name)
    subject = str(data["subject"])
    chest = data["signal"]["chest"]
    labels = np.asarray(data["label"]).reshape(-1)
    acc = np.asarray(chest["ACC"], dtype=float)
    ecg = np.asarray(chest["ECG"], dtype=float).reshape(-1)
    emg = np.asarray(chest["EMG"], dtype=float).reshape(-1)
    eda = np.asarray(chest["EDA"], dtype=float).reshape(-1)
    temp = np.asarray(chest["Temp"], dtype=float).reshape(-1)
    resp = np.asarray(chest["Resp"], dtype=float).reshape(-1)
    win = 14000
    step = 14000
    rows = []
    for start in range(0, len(labels) - win, step):
        end = start + win
        label = mode_label(labels[start:end])
        if label not in (1, 2, 3, 4):
            continue
        acc_part = acc[start:end]
        acc_mag = np.sqrt(np.sum(acc_part * acc_part, axis=1))
        ecg_part = ecg[start:end]
        emg_part = emg[start:end]
        eda_part = eda[start:end]
        temp_part = temp[start:end]
        resp_part = resp[start:end]
        rows.append(
            {
                "dataset": "WESAD",
                "subject": subject,
                "split": "",
                "age": float(subject.replace("S", "")) if subject.replace("S", "").isdigit() else 0.0,
                "sex": clean_number(acc_mag),
                "bmi": std_number(acc_mag),
                "bp": std_number(ecg_part),
                "s1": clean_number(eda_part),
                "s2": slope_number(eda_part),
                "s3": clean_number(temp_part),
                "s4": std_number(resp_part),
                "s5": std_number(emg_part),
                "s6": clean_number(np.abs(ecg_part)),
                "progression_score": float(label == 2),
                "high_risk": int(label == 2),
            }
        )
    return pd.DataFrame(rows)


def build_wesad():
    zip_path = RAW / "WESAD.zip"
    parts = []
    for name in wesad_members():
        parts.append(build_wesad_subject(zip_path, name))
    df = pd.concat(parts, ignore_index=True)
    subjects = sorted(df["subject"].unique())
    rng = np.random.default_rng(SEED)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * 0.2)))
    test_subjects = set(shuffled[:test_count])
    df["split"] = np.where(df["subject"].isin(test_subjects), "test", "train")
    return df, "stress state label equals 2", 0.5, "subject-level 80/20 split"


def ppg_members():
    with zipfile.ZipFile(RAW / "PPG_FieldStudy.zip") as z:
        names = z.namelist()
    return [name for name in names if name.endswith(".pkl") and "/S" in name]


def take_range(arr, start, end, ratio):
    a = int(round(start * ratio))
    b = int(round(end * ratio))
    return np.asarray(arr[a:b], dtype=float)


def build_ppg_subject(zip_path, name):
    data = read_pickle_from_zip(zip_path, name)
    subject = str(data["subject"])
    wrist = data["signal"]["wrist"]
    label = np.asarray(data["label"], dtype=float).reshape(-1)
    acc = np.asarray(wrist["ACC"], dtype=float)
    bvp = np.asarray(wrist["BVP"], dtype=float).reshape(-1)
    eda = np.asarray(wrist["EDA"], dtype=float).reshape(-1)
    temp = np.asarray(wrist["TEMP"], dtype=float).reshape(-1)
    win = 32
    step = 16
    rows = []
    for start in range(0, len(label) - win, step):
        end = start + win
        hr = clean_number(label[start:end])
        bvp_part = take_range(bvp, start, end, len(bvp) / len(label))
        acc_part = take_range(acc, start, end, len(acc) / len(label))
        eda_part = take_range(eda, start, end, len(eda) / len(label))
        temp_part = take_range(temp, start, end, len(temp) / len(label))
        if bvp_part.size == 0 or acc_part.size == 0:
            continue
        acc_mag = np.sqrt(np.sum(acc_part * acc_part, axis=1))
        rows.append(
            {
                "dataset": "PPG-DaLiA",
                "subject": subject,
                "split": "",
                "age": float(subject.replace("S", "")) if subject.replace("S", "").isdigit() else 0.0,
                "sex": clean_number(acc_mag),
                "bmi": std_number(acc_mag),
                "bp": clean_number(np.abs(bvp_part)),
                "s1": clean_number(bvp_part),
                "s2": std_number(bvp_part),
                "s3": clean_number(eda_part),
                "s4": clean_number(temp_part),
                "s5": std_number(acc_mag),
                "s6": slope_number(bvp_part),
                "progression_score": hr,
                "high_risk": 0,
            }
        )
    return pd.DataFrame(rows)


def build_ppg():
    zip_path = RAW / "PPG_FieldStudy.zip"
    parts = []
    for name in ppg_members():
        parts.append(build_ppg_subject(zip_path, name))
    df = pd.concat(parts, ignore_index=True)
    subjects = sorted(df["subject"].unique())
    rng = np.random.default_rng(SEED)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * 0.2)))
    test_subjects = set(shuffled[:test_count])
    df["split"] = np.where(df["subject"].isin(test_subjects), "test", "train")
    cut = float(df.loc[df["split"] == "train", "progression_score"].quantile(0.75))
    df["high_risk"] = (df["progression_score"] >= cut).astype(int)
    return df, "top 25 percent training heart-rate windows", cut, "subject-level 80/20 split"


class RulesOnlyBaseline(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def fit(self, x, y):
        self.columns_ = list(x.columns)
        self.classes_ = np.array([0, 1])
        self.levels_ = {}
        for col in ["bmi", "bp", "s5", "s6"]:
            self.levels_[col] = float(x[col].quantile(0.70))
        return self

    def score_rows(self, x):
        score = np.zeros(len(x), dtype=float)
        for col in ["bmi", "bp", "s5", "s6"]:
            score += (x[col].values > self.levels_[col]).astype(float)
        return np.clip(score / 4.0, 0.02, 0.98)

    def predict_proba(self, x):
        p = self.score_rows(x)
        return np.column_stack([1 - p, p])

    def predict(self, x):
        return (self.score_rows(x) >= 0.50).astype(int)


class HumanScheduleBaseline(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        self.age_ = float(x["age"].quantile(0.65))
        self.bmi_ = float(x["bmi"].quantile(0.65))
        self.bp_ = float(x["bp"].quantile(0.65))
        return self

    def score_rows(self, x):
        score = (
            0.34 * (x["age"].values > self.age_).astype(float)
            + 0.38 * (x["bmi"].values > self.bmi_).astype(float)
            + 0.28 * (x["bp"].values > self.bp_).astype(float)
        )
        return np.clip(score, 0.03, 0.97)

    def predict_proba(self, x):
        p = self.score_rows(x)
        return np.column_stack([1 - p, p])

    def predict(self, x):
        return (self.score_rows(x) >= 0.50).astype(int)


def models():
    base_lr = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        ]
    )
    base_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=SEED,
    )
    base_gb = GradientBoostingClassifier(random_state=SEED)
    base_et = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
    )
    base_svm = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.2, probability=True, class_weight="balanced", random_state=SEED)),
        ]
    )
    return {
        "Rules-only (B1)": RulesOnlyBaseline(),
        "Predictive-only (B2)": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "Human-schedule (B3)": HumanScheduleBaseline(),
        "AgHealth+": VotingClassifier(
            estimators=[
                ("lr", base_lr),
                ("rf", base_rf),
                ("gb", base_gb),
                ("et", base_et),
                ("svm", base_svm),
            ],
            voting="soft",
            weights=[3, 2, 2, 2, 2],
        ),
    }


def get_score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def train_all(model_map, x_train, y_train):
    fitted = {}
    for name, model in model_map.items():
        model.fit(x_train, y_train)
        fitted[name] = model
    return fitted


def measure_all(fitted, x_test, y_test):
    rows = []
    pred_store = {}
    for name, model in fitted.items():
        pred = model.predict(x_test)
        prob = get_score(model, x_test)
        pred_store[name] = {"pred": pred, "prob": prob}
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False), pred_store


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": BLACK,
            "axes.linewidth": 1.4,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "font.size": 10,
            "legend.frameon": True,
            "legend.edgecolor": BLACK,
            "savefig.facecolor": "white",
        }
    )


def frame(ax):
    for item in ax.spines.values():
        item.set_visible(True)
        item.set_color(BLACK)
        item.set_linewidth(1.4)
    ax.tick_params(width=1.2, color=BLACK)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_target(df, slug, title):
    counts = df["high_risk"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.bar(["Low risk", "High risk"], counts.values, color=[BLUE, ORANGE], edgecolor=BLACK, linewidth=1.4)
    ax.set_title(f"{title} Target Distribution")
    ax.set_ylabel("Windows")
    frame(ax)
    save_fig(f"{slug}_target_distribution")


def plot_roc(pred_store, y_test, slug, title):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for i, (name, item) in enumerate(pred_store.items()):
        fpr, tpr, _ = roc_curve(y_test, item["prob"])
        auc = roc_auc_score(y_test, item["prob"])
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=2.4, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color=BLACK, linewidth=1.1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} ROC Curves")
    ax.legend(fontsize=8, facecolor="white")
    frame(ax)
    save_fig(f"{slug}_roc_curves")


def plot_matrix(y_test, pred_store, name, slug, title):
    cm = confusion_matrix(y_test, pred_store[name]["pred"])
    fig, ax = plt.subplots(figsize=(5.8, 5.1))
    colors = np.array([[0, 1], [2, 3]])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([BLUE, ORANGE, RED, GREEN])
    ax.imshow(colors, cmap=cmap, vmin=0, vmax=3)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=BLACK, fontsize=18)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low", "High"])
    ax.set_yticklabels(["Low", "High"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color=BLACK, linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(f"{title} Confusion Matrix - {name}")
    frame(ax)
    save_fig(f"{slug}_confusion_matrix_best")


def feature_values(model, columns):
    if hasattr(model, "estimators_"):
        parts = []
        for est in model.estimators_:
            item = feature_values(est, columns)
            total = float(item["value"].sum())
            if total > 0:
                item["value"] = item["value"] / total
                parts.append(item["value"].values)
        if parts:
            vals = np.mean(np.vstack(parts), axis=0)
            return pd.DataFrame({"feature": columns, "value": vals})
    raw = model
    if hasattr(model, "named_steps"):
        raw = model.named_steps["model"]
    if hasattr(raw, "feature_importances_"):
        return pd.DataFrame({"feature": columns, "value": raw.feature_importances_})
    if hasattr(raw, "coef_"):
        return pd.DataFrame({"feature": columns, "value": np.abs(raw.coef_[0])})
    if hasattr(raw, "theta_") and hasattr(raw, "var_"):
        diff = np.abs(raw.theta_[1] - raw.theta_[0])
        scale = np.sqrt(np.maximum(raw.var_[0] + raw.var_[1], 1e-12))
        return pd.DataFrame({"feature": columns, "value": diff / scale})
    return pd.DataFrame({"feature": columns, "value": np.zeros(len(columns))})


def plot_features(model, columns, slug, title):
    imp = feature_values(model, columns).sort_values("value", ascending=True)
    imp.to_csv(OUT / f"{slug}_feature_importance.csv", index=False)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(imp))]
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.barh(imp["feature"], imp["value"], color=colors, edgecolor=BLACK, linewidth=1.1)
    ax.set_title(f"{title} Feature Importance")
    ax.set_xlabel("Importance")
    frame(ax)
    save_fig(f"{slug}_feature_importance")


def plot_metrics(metrics, slug, title):
    show = metrics.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(show.index))
    width = 0.14
    for i, col in enumerate(show.columns):
        ax.bar(
            x + (i - 2) * width,
            show[col].values,
            width,
            label=col.upper(),
            color=PALETTE[i],
            edgecolor=BLACK,
            linewidth=0.9,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{title} Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(show.index, rotation=18, ha="right")
    ax.legend(ncol=5, fontsize=8, facecolor="white")
    frame(ax)
    save_fig(f"{slug}_model_comparison")


def markdown_table(df):
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, item in df.iterrows():
        vals = []
        for col in cols:
            val = item[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def save_dataset_files(df, x_train, x_test, y_train, y_test, slug):
    df.to_csv(DATA_DIR / f"{slug}_dataset.csv", index=False)
    train = x_train.copy()
    test = x_test.copy()
    train["high_risk"] = y_train.values
    test["high_risk"] = y_test.values
    train.to_csv(DATA_DIR / f"{slug}_train_split.csv", index=False)
    test.to_csv(DATA_DIR / f"{slug}_test_split.csv", index=False)


def save_predictions(x_test, y_test, pred_store, name, slug):
    pred = x_test.copy()
    pred["true_high_risk"] = y_test.values
    pred["predicted_high_risk"] = pred_store[name]["pred"]
    pred["risk_probability"] = pred_store[name]["prob"]
    pred.to_csv(OUT / f"{slug}_best_model_predictions.csv", index=False)


def save_confusion(y_test, pred_store, name, slug):
    cm = confusion_matrix(y_test, pred_store[name]["pred"])
    table = pd.DataFrame(
        [
            {"cell": "true_low_pred_low", "count": int(cm[0, 0])},
            {"cell": "true_low_pred_high", "count": int(cm[0, 1])},
            {"cell": "true_high_pred_low", "count": int(cm[1, 0])},
            {"cell": "true_high_pred_high", "count": int(cm[1, 1])},
        ]
    )
    table.to_csv(OUT / f"{slug}_confusion_matrix_counts.csv", index=False)


def evaluate_dataset(df, title, slug, target_rule, cutoff, split_note):
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["high_risk"])
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    x_train = train[FEATURES]
    y_train = train["high_risk"].astype(int)
    x_test = test[FEATURES]
    y_test = test["high_risk"].astype(int)
    fitted = train_all(models(), x_train, y_train)
    metrics, pred_store = measure_all(fitted, x_test, y_test)
    best = metrics.iloc[0]["model"]
    save_dataset_files(df, x_train, x_test, y_train, y_test, slug)
    metrics.to_csv(OUT / f"{slug}_model_metrics.csv", index=False)
    save_predictions(x_test, y_test, pred_store, best, slug)
    save_confusion(y_test, pred_store, best, slug)
    plot_target(df, slug, title)
    plot_metrics(metrics, slug, title)
    plot_roc(pred_store, y_test, slug, title)
    plot_matrix(y_test, pred_store, best, slug, title)
    plot_features(fitted[best], FEATURES, slug, title)
    row = metrics[metrics["model"] == best].iloc[0].to_dict()
    summary = {
        "dataset": title,
        "slug": slug,
        "rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": len(FEATURES),
        "target_rule": target_rule,
        "cutoff": float(cutoff),
        "split": split_note,
        "best_model": best,
        "best_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in row.items()},
    }
    with open(OUT / f"{slug}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    lines = [
        f"# {title} Results",
        "",
        f"Dataset: {title}",
        f"Rows: {summary['rows']}",
        f"Train rows: {summary['train_rows']}",
        f"Test rows: {summary['test_rows']}",
        f"Features: {summary['features']}",
        f"Target: {target_rule}",
        f"Cutoff: {summary['cutoff']:.4f}",
        f"Split: {split_note}",
        f"Best model: {best}",
        "",
        markdown_table(metrics),
    ]
    (OUT / f"{slug}_results_summary.md").write_text("\n".join(lines))
    return metrics.assign(dataset=title), summary


def save_config(summaries):
    config = {
        "datasets": [item["dataset"] for item in summaries],
        "target": "high_risk",
        "features": FEATURES,
        "random_seed": SEED,
        "methods": METHODS,
        "model_setup": "same practical paper model implementation",
    }
    with open(CONFIG_DIR / "three_dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_docs(all_metrics, summaries):
    table = all_metrics[["dataset", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    best_rows = table.sort_values(["dataset", "roc_auc"], ascending=[True, False]).groupby("dataset").head(1)
    lines = [
        "# Paper Replacement Numbers",
        "",
        "The same four methods were evaluated on three real datasets.",
        "",
        "## Best Models",
        "",
        markdown_table(best_rows.reset_index(drop=True)),
        "",
        "## All Results",
        "",
        markdown_table(table.reset_index(drop=True)),
        "",
        "## Dataset Notes",
        "",
    ]
    for item in summaries:
        lines.extend(
            [
                f"### {item['dataset']}",
                "",
                f"Rows: {item['rows']}",
                f"Train rows: {item['train_rows']}",
                f"Test rows: {item['test_rows']}",
                f"Target rule: {item['target_rule']}",
                f"Split: {item['split']}",
                "",
            ]
        )
    (DOCS_DIR / "paper_replacement_numbers.md").write_text("\n".join(lines))
    (OUT / "model_metrics.csv").write_text(table.to_csv(index=False))
    (OUT / "results_summary.md").write_text("\n".join(lines))


def sync_root_copies():
    root_fig = ROOT / "figures"
    root_out = ROOT / "outputs"
    root_fig.mkdir(exist_ok=True)
    root_out.mkdir(exist_ok=True)
    for item in FIG.glob("*"):
        if item.is_file():
            shutil.copy2(item, root_fig / item.name)
    for item in OUT.glob("*"):
        if item.is_file():
            shutil.copy2(item, root_out / item.name)
    for item in DATA_DIR.glob("*.csv"):
        shutil.copy2(item, root_out / item.name)


def build_all():
    builders = [
        ("OhioT1DM", "ohiot1dm", build_ohio),
        ("WESAD", "wesad", build_wesad),
        ("PPG-DaLiA", "ppg_dalia", build_ppg),
    ]
    frames = []
    summaries = []
    for title, slug, builder in builders:
        df, rule, cut, split_note = builder()
        metrics, summary = evaluate_dataset(df, title, slug, rule, cut, split_note)
        frames.append(metrics)
        summaries.append(summary)
    all_metrics = pd.concat(frames, ignore_index=True)
    save_config(summaries)
    save_docs(all_metrics, summaries)
    sync_root_copies()
    return all_metrics, summaries


def run():
    make_dirs()
    setup_style()
    all_metrics, summaries = build_all()
    print(all_metrics[["dataset", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    for item in summaries:
        print(f"{item['dataset']} best_model={item['best_model']}")


if __name__ == "__main__":
    run()
