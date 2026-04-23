"""
Figure 3 — Calibration curves, all six models after Platt scaling
Two panels: left = T-stage, right = Surgery
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

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

TSTAGE_MODELS = [
    {"label": "Clinical only", "file": "oof_Clinical_only.csv",
     "color": COL_CLINICAL, "ls": "-",  "marker": "o"},
    {"label": "Imaging only",  "file": "oof_Imaging_only.csv",
     "color": COL_IMAGING,  "ls": "-",  "marker": "s"},
    {"label": "Fusion",        "file": "oof_Fusion.csv",
     "color": COL_FUSION,   "ls": "--", "marker": "^"},
]

SURGERY_MODELS = [
    {"label": "Clinical only", "file": "oof_surgery_Clinical_A_no_neoadj.csv",
     "color": COL_CLINICAL, "ls": "-",  "marker": "o"},
    {"label": "Imaging only",  "file": "oof_surgery_Imaging_A_no_neoadj.csv",
     "color": COL_IMAGING,  "ls": "-",  "marker": "s"},
    {"label": "Fusion",        "file": "oof_surgery_Fusion_A_no_neoadj.csv",
     "color": COL_FUSION,   "ls": "--", "marker": "^"},
]


def platt_scale(y_true, y_pred):
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    return cross_val_predict(
        lr, y_pred.reshape(-1,1), y_true,
        cv=10, method="predict_proba")[:,1]


def smooth_curve(y_true, y_pred, n=300):
    ir = IsotonicRegression(out_of_bounds="clip").fit(y_pred, y_true)
    x  = np.linspace(y_pred.min(), y_pred.max(), n)
    return x, ir.predict(x)


def ece_score(y_true, y_pred, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece  = 0.0
    for i in range(n_bins):
        lo,hi = bins[i],bins[i+1]
        mask  = (y_pred>=lo)&(y_pred<=(hi if i==n_bins-1 else y_pred<hi))
        mask  = (y_pred>=lo)&(y_pred<hi) if i<n_bins-1 else (y_pred>=lo)&(y_pred<=hi)
        n = mask.sum()
        if n==0: continue
        ece += n/len(y_true)*abs(y_pred[mask].mean()-y_true[mask].mean())
    return ece


def plot_panel(ax, models, title):
    ax.plot([0,1],[0,1], color=COL_GREY, lw=1.2,
            linestyle="--", label="Perfect calibration", zorder=1)

    for m in models:
        path = OUTPUT_DIR / m["file"]
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue

        df     = pd.read_csv(path)
        y_true = df["y_true"].values.astype(int)
        y_pred = df["y_pred"].values.astype(float)
        y_cal  = platt_scale(y_true, y_pred)
        ece    = ece_score(y_true, y_cal)

        # Smooth calibration curve
        x, y = smooth_curve(y_true, y_cal)
        ax.plot(x, y, color=m["color"], lw=2.2,
                linestyle=m["ls"], zorder=3,
                label=f"{m['label']}  (ECE={ece:.3f})")

        # Reliability dots
        bins = np.linspace(0,1,11)
        mp_list, op_list, cnt_list = [], [], []
        for i in range(10):
            lo,hi = bins[i],bins[i+1]
            mask  = (y_cal>=lo)&(y_cal<=hi if i==9 else y_cal<hi)
            if mask.sum() < 5: continue
            mp_list.append(y_cal[mask].mean())
            op_list.append(y_true[mask].mean())
            cnt_list.append(mask.sum())

        ax.scatter(mp_list, op_list,
                   color=m["color"], marker=m["marker"],
                   s=np.array(cnt_list)*0.5+12,
                   alpha=0.7, zorder=4)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Observed frequency", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8, fontweight="500")
    ax.legend(fontsize=8.5, loc="upper left",
              framealpha=0.95, edgecolor="#D3D1C7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)


fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13, 5.6),
    gridspec_kw={"wspace": 0.30}
)
fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.11)

fig.suptitle("Calibration curves — after Platt scaling",
             fontsize=14, fontweight="500", y=0.99)

plot_panel(ax1, TSTAGE_MODELS,
           "T-stage prediction  (n=916)")

plot_panel(ax2, SURGERY_MODELS,
           "Surgical planning  (mastectomy vs BCS,  n=559)")

# Panel labels
for ax, lbl in [(ax1,"A"), (ax2,"B")]:
    ax.text(-0.08, 1.04, lbl, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

# Shared legend note
fig.text(0.5, 0.01,
         "Point size proportional to bin count.  "
         "Dashed line = perfect calibration.",
         ha="center", fontsize=9, color="#888780")

out = OUTPUT_DIR / "poster_fig3_calibration_final.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
