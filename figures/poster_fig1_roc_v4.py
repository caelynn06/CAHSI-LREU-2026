"""
Figure 1 — ROC curves v4
  - Shading under winning curve per panel
  - Cohort A label removed
  - Clean titles
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("pipeline_outputs")

COL_CLINICAL = "#3D52A0"
COL_IMAGING  = "#1D9E75"
COL_FUSION   = "#D85A30"
COL_GREY     = "#B4B2A9"


def load_oof(path):
    df = pd.read_csv(path)
    return df["y_true"].values.astype(int), df["y_pred"].values.astype(float)


def bootstrap_ci(y_true, y_pred, n=1000, seed=42):
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def plot_panel(ax, models, title, shade_label):
    """
    shade_label: label of the model to shade under (the winner)
    """
    ax.plot([0,1],[0,1], color=COL_GREY, lw=1.2,
            linestyle="--", label="Random classifier", zorder=1)

    curves = {}
    for m in models:
        path = OUTPUT_DIR / m["oof_path"]
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping")
            continue
        y_true, y_pred = load_oof(path)
        auc = roc_auc_score(y_true, y_pred)
        lo, hi = bootstrap_ci(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        curves[m["label"]] = (fpr, tpr, m["color"])

        label = (f"{m['label']}\n"
                 f"AUC = {auc:.3f} [{lo:.3f}–{hi:.3f}]")
        ax.plot(fpr, tpr, color=m["color"], lw=2.2,
                linestyle=m.get("ls", "-"), label=label, zorder=3)

    # Shade under the winning model only
    if shade_label in curves:
        fpr, tpr, col = curves[shade_label]
        ax.fill_between(fpr, tpr, alpha=0.12, color=col, zorder=2)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("1 – Specificity (FPR)", fontsize=12)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax.set_title(title, fontsize=12, pad=8, fontweight="500")
    ax.legend(fontsize=9, loc="lower right",
              framealpha=0.95, edgecolor="#D3D1C7",
              handlelength=1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(labelsize=10)
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.4,
            alpha=0.45, color="#cccccc")


TSTAGE_MODELS = [
    {"label": "Clinical only",
     "oof_path": "oof_Clinical_only.csv",
     "color": COL_CLINICAL, "ls": "-"},
    {"label": "Imaging only",
     "oof_path": "oof_Imaging_only.csv",
     "color": COL_IMAGING, "ls": "-"},
    {"label": "Fusion",
     "oof_path": "oof_Fusion.csv",
     "color": COL_FUSION, "ls": "--"},
]

SURGERY_MODELS = [
    {"label": "Clinical only",
     "oof_path": "oof_surgery_Clinical_A_no_neoadj.csv",
     "color": COL_CLINICAL, "ls": "-"},
    {"label": "Imaging only",
     "oof_path": "oof_surgery_Imaging_A_no_neoadj.csv",
     "color": COL_IMAGING, "ls": "-"},
    {"label": "Fusion",
     "oof_path": "oof_surgery_Fusion_A_no_neoadj.csv",
     "color": COL_FUSION, "ls": "--"},
]

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13, 5.6),
    gridspec_kw={"wspace": 0.30}
)
fig.subplots_adjust(left=0.07, right=0.97,
                    top=0.91, bottom=0.11)

fig.suptitle("ROC curves — clinical, imaging, and fusion models",
             fontsize=14, fontweight="500", y=0.99)

# T-stage: imaging wins → shade imaging
plot_panel(
    ax1, TSTAGE_MODELS,
    "T-stage prediction  (T0/T1 vs T2+,  n=916)",
    shade_label="Imaging only"
)

# Surgery: clinical wins → shade clinical
plot_panel(
    ax2, SURGERY_MODELS,
    "Surgical planning  (mastectomy vs BCS,  n=559)",
    shade_label="Clinical only"
)

# Panel labels
for ax, lbl in [(ax1, "A"), (ax2, "B")]:
    ax.text(-0.08, 1.04, lbl, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

out = OUTPUT_DIR / "poster_fig1_roc_final.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
