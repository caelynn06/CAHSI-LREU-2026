"""
Figure 4 — Performance summary table (all three models, all metrics)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("pipeline_outputs")

# (Target, Model, N, AUC, CI_lo, CI_hi, ECE, Brier)
ROWS = [
    ("T-stage (T0/T1 vs T2+)", "Clinical only", 916, 0.608, 0.572, 0.644, 0.019, 0.239),
    ("T-stage (T0/T1 vs T2+)", "Imaging only",  916, 0.787, 0.760, 0.819, 0.025, 0.187),
    ("T-stage (T0/T1 vs T2+)", "Fusion",         916, 0.788, 0.759, 0.817, 0.048, 0.190),
    ("Surgical planning\n(mastectomy vs BCS,\nn=559)", "Clinical only", 559, 0.714, 0.667, 0.757, 0.056, 0.207),
    ("Surgical planning\n(mastectomy vs BCS,\nn=559)", "Imaging only",  559, 0.668, 0.623, 0.715, 0.032, 0.224),
    ("Surgical planning\n(mastectomy vs BCS,\nn=559)", "Fusion",         559, 0.731, 0.683, 0.776, 0.046, 0.203),
]

COLS       = ["Target / cohort", "Model", "N", "AUC", "95% CI", "ECE*", "Brier*"]
COL_WIDTHS = [0.23, 0.13, 0.06, 0.07, 0.16, 0.07, 0.07]
HEADER_COL = "#3D52A0"
ROW_COLS   = ["#F5F5F8", "#EEEEF4"]

# Highest AUC per group: T-stage = Fusion row 3, Surgery = Fusion row 6
BEST_ROWS = {3, 6}

fig, ax = plt.subplots(figsize=(12, 5.2))
ax.axis("off")

cell_text = []
for r in ROWS:
    target, model, n, auc, lo, hi, ece, brier = r
    cell_text.append([
        target, model, str(n),
        f"{auc:.3f}",
        f"[{lo:.3f}, {hi:.3f}]",
        f"{ece:.3f}",
        f"{brier:.3f}",
    ])

table = ax.table(
    cellText=cell_text,
    colLabels=COLS,
    cellLoc="center",
    loc="center",
    colWidths=COL_WIDTHS,
)
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 2.5)

# Header
for j in range(len(COLS)):
    cell = table[0, j]
    cell.set_facecolor(HEADER_COL)
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("white")

# Row colours
rc_map = {1: ROW_COLS[0], 2: ROW_COLS[0], 3: ROW_COLS[0],
          4: ROW_COLS[1], 5: ROW_COLS[1], 6: ROW_COLS[1]}

for i in range(1, len(ROWS) + 1):
    for j in range(len(COLS)):
        cell = table[i, j]
        cell.set_facecolor(rc_map[i])
        cell.set_edgecolor("white")
        cell.set_text_props(color="#2C2C2A")

# Bold highest AUC per target
for row_idx in BEST_ROWS:
    table[row_idx, 3].set_text_props(fontweight="bold", color="#2C2C2A")

# Left-align first two columns
for i in range(1, len(ROWS) + 1):
    table[i, 0].set_text_props(ha="left")
    table[i, 1].set_text_props(ha="left")

# Hide repeated target labels
for i in [2, 3, 5, 6]:
    table[i, 0].set_text_props(color=rc_map[i])

ax.set_title("Model performance summary",
             fontsize=13, fontweight="500",
             pad=10, loc="left", x=0.01)

fig.text(0.01, 0.02,
         "* After Platt scaling.  "
         "ECE = Expected Calibration Error (lower = better).  "
         "CI = 95% bootstrap confidence interval.  "
         "Bold AUC = highest per target.",
         fontsize=8.5, color="#5F5E5A")

plt.tight_layout()
out = OUTPUT_DIR / "poster_fig4_table_final.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
