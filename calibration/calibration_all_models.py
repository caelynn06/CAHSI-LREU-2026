"""
Calibration — All Six Models
=============================
Runs Platt scaling on all OOF prediction files and reports
ECE and Brier score for every model.

Outputs
-------
  pipeline_outputs/calibration_all_models.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("pipeline_outputs")

MODELS = [
    # T-stage
    {"label": "T-stage — Clinical only", "file": "oof_Clinical_only.csv"},
    {"label": "T-stage — Imaging only",  "file": "oof_Imaging_only.csv"},
    {"label": "T-stage — Fusion",        "file": "oof_Fusion.csv"},
    # Surgery Cohort A
    {"label": "Surgery — Clinical only", "file": "oof_surgery_Clinical_A_no_neoadj.csv"},
    {"label": "Surgery — Imaging only",  "file": "oof_surgery_Imaging_A_no_neoadj.csv"},
    {"label": "Surgery — Fusion",        "file": "oof_surgery_Fusion_A_no_neoadj.csv"},
]


def platt_scale(y_true, y_pred):
    lr  = LogisticRegression(C=1.0, solver="lbfgs")
    cal = cross_val_predict(lr, y_pred.reshape(-1,1), y_true,
                            cv=10, method="predict_proba")[:,1]
    return cal


def ece_score(y_true, y_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_pred >= lo) & (y_pred <= hi if i==n_bins-1 else y_pred < hi)
        n = mask.sum()
        if n == 0:
            continue
        ece += n / len(y_true) * abs(y_pred[mask].mean() - y_true[mask].mean())
    return ece


print("=" * 60)
print("Calibration — All Models")
print("=" * 60)

rows = []
for m in MODELS:
    path = OUTPUT_DIR / m["file"]
    if not path.exists():
        print(f"\n  SKIPPING {m['label']} — file not found")
        continue

    df     = pd.read_csv(path)
    y_true = df["y_true"].values.astype(int)
    y_pred = df["y_pred"].values.astype(float)

    y_cal   = platt_scale(y_true, y_pred)
    auc     = roc_auc_score(y_true, y_pred)
    ece_raw = ece_score(y_true, y_pred)
    ece_cal = ece_score(y_true, y_cal)
    brier   = brier_score_loss(y_true, y_cal)

    print(f"\n  {m['label']}")
    print(f"    AUC        : {auc:.4f}")
    print(f"    ECE before : {ece_raw:.4f}")
    print(f"    ECE after  : {ece_cal:.4f}")
    print(f"    Brier      : {brier:.4f}")

    rows.append({
        "model":       m["label"],
        "auc":         auc,
        "ece_before":  ece_raw,
        "ece_after":   ece_cal,
        "brier":       brier,
    })

df_out = pd.DataFrame(rows)
out    = OUTPUT_DIR / "calibration_all_models.csv"
df_out.to_csv(out, index=False)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Model':<30} {'AUC':>6}  {'ECE before':>10}  {'ECE after':>9}  {'Brier':>6}")
print("-"*68)
for _, r in df_out.iterrows():
    print(f"{r['model']:<30} {r['auc']:>6.4f}  "
          f"{r['ece_before']:>10.4f}  {r['ece_after']:>9.4f}  "
          f"{r['brier']:>6.4f}")

print(f"\nSaved: {out}")
