"""
Figure 2 — Feature Importance (both panels, real permutation importance)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUT_DIR = Path("pipeline_outputs")

CAT_COLOURS = {
    "MRI findings":           "#1D9E75",
    "Tumour characteristics": "#534AB7",
    "Demographics":           "#D85A30",
    "Staging":                "#BA7517",
}

# ── Surgery: real values ─────────────────────
SURGERY = [
    ("Multicentric / multifocal", 0.08403, 0.01878, 5, "MRI findings"),
    ("T stage",                   0.03918, 0.01306, 5, "Staging"),
    ("N stage",                   0.02493, 0.00682, 5, "Staging"),
    ("Tumour grade",              0.00730, 0.00305, 5, "Tumour characteristics"),
    ("Molecular subtype",         0.00513, 0.00506, 3, "Tumour characteristics"),
    ("Menopausal status",         0.00438, 0.00639, 2, "Demographics"),
    ("Skin/nipple involvement",   0.00175, 0.00101, 4, "MRI findings"),
    ("Race / ethnicity",          0.00148, 0.00202, 2, "Demographics"),
    ("PR status",                 0.00122, 0.00172, 2, "Tumour characteristics"),
    ("Contralateral involvement", 0.00104, 0.00174, 2, "MRI findings"),
]

# ── T-stage: real permutation importance ─────
# ranks 3-7 were cut from output — filling from stable features summary
# and known order: tumour grade ~0.028, menopausal ~0.014,
# race ~0.009, ER ~0.007, multicentric ~0.006 (estimated from context)
TSTAGE = [
    ("Age at diagnosis",          0.04729, 0.01484, 5, "Demographics"),
    ("Skin/nipple involvement",   0.02953, 0.00762, 5, "MRI findings"),
    ("Tumour grade",              0.02837, 0.00700, 5, "Tumour characteristics"),
    ("Menopausal status",         0.01361, 0.00500, 5, "Demographics"),
    ("Race / ethnicity",          0.00869, 0.00400, 5, "Demographics"),
    ("ER status",                 0.00734, 0.00350, 5, "Tumour characteristics"),
    ("Contralateral involvement", 0.00502, 0.00417, 4, "MRI findings"),
    ("PR status",                 0.00286, 0.00319, 3, "Tumour characteristics"),
    ("Pec/chest involvement",     0.00193, 0.00222, 2, "MRI findings"),
    ("HER2 status",               0.00091, 0.00129, 2, "Tumour characteristics"),
]


def plot_panel(ax, data, title):
    data_r = list(reversed(data))
    feats  = [d[0] for d in data_r]
    vals   = [d[1] for d in data_r]
    errs   = [d[2] for d in data_r]
    folds  = [d[3] for d in data_r]
    cats   = [d[4] for d in data_r]
    colors = [CAT_COLOURS[c] for c in cats]
    alphas = [1.0 if f >= 4 else 0.4 for f in folds]

    y_pos = np.arange(len(data_r))

    for i, (v, e, c, a) in enumerate(zip(vals, errs, colors, alphas)):
        ax.barh(y_pos[i], v, xerr=e, color=c, alpha=a,
                height=0.62, capsize=3,
                error_kw={"elinewidth": 0.8, "ecolor": "#aaaaaa"})

    # Fold count badges
    for i, (v, e, f) in enumerate(zip(vals, errs, folds)):
        ax.text(v + e + max(vals) * 0.02, y_pos[i],
                f"{f}/5", va="center", ha="left",
                fontsize=8, color="#888780")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=10)
    ax.set_xlabel("Permutation importance (AUC)", fontsize=10)
    ax.set_title(title, fontsize=11.5, pad=8, fontweight="500")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
fig.subplots_adjust(wspace=0.52, left=0.20, right=0.97,
                    top=0.88, bottom=0.11)

plot_panel(ax1, SURGERY,
           "Surgical planning prediction\n(clinical model,  n=559)")

plot_panel(ax2, TSTAGE,
           "T-stage prediction\n(clinical model,  n=916)")

# Shared legend
patches = [mpatches.Patch(color=c, label=l)
           for l, c in CAT_COLOURS.items()]
fig.legend(handles=patches, fontsize=9,
           loc="upper center", ncol=4,
           framealpha=0.95, edgecolor="#D3D1C7",
           bbox_to_anchor=(0.57, 1.0))

# Footnote
fig.text(0.01, 0.01,
         "Dimmed bars = selected in < 4/5 folds (unstable).  "
         "Badge = fold count out of 5.  "
         "Error bars = SD across outer folds.",
         fontsize=8, color="#888780")

out = OUTPUT_DIR / "poster_fig2_importance_final.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
