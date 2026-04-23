"""
Figure 5 — Prediction probability distributions v2
====================================================
Clean overlapping KDE density curves only — no scatter.
Four panels: T-stage clinical, T-stage imaging,
             Surgery clinical, Surgery imaging
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("pipeline_outputs")

COL_EARLY = "#4A90D9"
COL_LATE  = "#D85A30"

PANELS = [
    {"file": "oof_Clinical_only.csv",
     "title": "T-stage — clinical model",
     "label_0": "Early (T0/T1)", "label_1": "Late (T2+)", "panel": "A"},
    {"file": "oof_Imaging_only.csv",
     "title": "T-stage — imaging model",
     "label_0": "Early (T0/T1)", "label_1": "Late (T2+)", "panel": "B"},
    {"file": "oof_surgery_Clinical_A_no_neoadj.csv",
     "title": "Surgical planning — clinical model",
     "label_0": "BCS", "label_1": "Mastectomy", "panel": "C"},
    {"file": "oof_surgery_Imaging_A_no_neoadj.csv",
     "title": "Surgical planning — imaging model",
     "label_0": "BCS", "label_1": "Mastectomy", "panel": "D"},
]


def smooth_density(values, n_bins=80, sigma=2.5):
    counts, edges = np.histogram(values, bins=n_bins,
                                 range=(0, 1), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)
    return centers, smoothed


def plot_panel(ax, y_true, y_pred, title,
               label_0, label_1, panel_lbl):
    auc  = roc_auc_score(y_true, y_pred)
    idx0 = y_true == 0
    idx1 = y_true == 1

    x0, d0 = smooth_density(y_pred[idx0])
    x1, d1 = smooth_density(y_pred[idx1])

    # Normalise so both curves peak at 1 for fair comparison
    d0 = d0 / d0.max()
    d1 = d1 / d1.max()

    ax.fill_between(x0, d0, alpha=0.25, color=COL_EARLY)
    ax.fill_between(x1, d1, alpha=0.25, color=COL_LATE)
    ax.plot(x0, d0, color=COL_EARLY, lw=2.2,
            label=f"{label_0}  (n={idx0.sum()})")
    ax.plot(x1, d1, color=COL_LATE,  lw=2.2,
            label=f"{label_1}  (n={idx1.sum()})")

    # Decision threshold
    ax.axvline(0.5, color="#B4B2A9", lw=1.0,
               linestyle="--", alpha=0.8)

    # AUC box
    ax.text(0.97, 0.95, f"AUC = {auc:.3f}",
            transform=ax.transAxes,
            fontsize=10.5, ha="right", va="top",
            fontweight="500", color="#2C2C2A",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white",
                      edgecolor="#D3D1C7",
                      linewidth=0.5))

    # Overlap shading annotation
    overlap = np.minimum(d0, d1)
    ax.fill_between(x0, overlap, alpha=0.15,
                    color="#888780", zorder=0)

    # Panel label
    ax.text(-0.06, 1.06, panel_lbl,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Predicted probability", fontsize=10)
    ax.set_ylabel("Normalised density", fontsize=10)
    ax.set_title(title, fontsize=11.5, pad=6, fontweight="500")
    ax.legend(fontsize=9, loc="upper left",
              framealpha=0.95, edgecolor="#D3D1C7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.grid(axis="x", linestyle="--",
            linewidth=0.4, alpha=0.45, color="#cccccc")


fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
fig.subplots_adjust(hspace=0.45, wspace=0.28,
                    left=0.07, right=0.97,
                    top=0.91, bottom=0.09)

fig.suptitle("Predicted probability distributions by true outcome",
             fontsize=14, fontweight="500", y=0.98)

for ax, panel in zip(axes.flat, PANELS):
    path = OUTPUT_DIR / panel["file"]
    if not path.exists():
        print(f"  WARNING: {path} not found")
        ax.axis("off")
        continue
    df     = pd.read_csv(path)
    y_true = df["y_true"].values.astype(int)
    y_pred = df["y_pred"].values.astype(float)
    plot_panel(ax, y_true, y_pred,
               panel["title"], panel["label_0"],
               panel["label_1"], panel["panel"])

fig.text(0.5, 0.01,
         "Curves show smoothed predicted probability density per class.  "
         "Grey shading = overlap region.  "
         "Dashed line = 0.5 decision threshold.  "
         "Better separation = higher AUC.",
         ha="center", fontsize=8.5, color="#888780")

out = OUTPUT_DIR / "poster_fig5_distributions.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
