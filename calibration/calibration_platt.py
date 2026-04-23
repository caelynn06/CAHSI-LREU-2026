"""
Calibration with Platt Scaling
================================
Loads OOF predictions, applies Platt scaling (logistic regression
on OOF scores), then re-evaluates and plots calibration curves.

Platt scaling is fit on the OOF predictions themselves using
cross-validation to avoid overfitting the calibration.

Outputs
-------
  pipeline_outputs/figure_calibration_platt.png
  pipeline_outputs/calibration_metrics_platt.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("pipeline_outputs")

MODELS = [
    {"label": "Clinical — surgery (Cohort A)",
     "file":  OUTPUT_DIR / "oof_surgery_Clinical_A_no_neoadj.csv",
     "color": "#534AB7", "marker": "o"},
    {"label": "Fusion — surgery (Cohort A)",
     "file":  OUTPUT_DIR / "oof_surgery_Fusion_A_no_neoadj.csv",
     "color": "#D85A30", "marker": "s"},
    {"label": "Clinical — T-stage",
     "file":  OUTPUT_DIR / "oof_Clinical_only.csv",
     "color": "#888780", "marker": "^"},
]

N_BINS = 10


def platt_scale(y_true, y_pred):
    """
    Fit logistic regression on raw OOF scores to recalibrate.
    Uses cross_val_predict on the OOF set itself to avoid
    overfitting the calibration transform.
    """
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    X  = y_pred.reshape(-1, 1)
    # 10-fold CV on the OOF predictions for honest calibration
    calibrated = cross_val_predict(
        lr, X, y_true,
        cv=10, method="predict_proba"
    )[:, 1]
    return calibrated


def calibration_bins(y_true, y_prob, n_bins=10):
    bins, rows = np.linspace(0, 1, n_bins + 1), []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob <= hi if i==n_bins-1 else y_prob < hi)
        n = mask.sum()
        rows.append({
            "bin_mid":   (lo+hi)/2,
            "mean_pred": float(y_prob[mask].mean()) if n else np.nan,
            "frac_pos":  float(y_true[mask].mean()) if n else np.nan,
            "count":     int(n),
        })
    return pd.DataFrame(rows)


def ece_score(bin_df, n):
    v = bin_df.dropna(subset=["mean_pred","frac_pos"])
    if v.empty: return np.nan
    return float((v["count"]/n * (v["mean_pred"]-v["frac_pos"]).abs()).sum())


def iso_curve(y_true, y_prob):
    ir = IsotonicRegression(out_of_bounds="clip").fit(y_prob, y_true)
    x  = np.linspace(y_prob.min(), y_prob.max(), 300)
    return x, ir.predict(x)


def plot_calibration_comparison(models_data, outpath):
    """
    Three-panel figure per model:
      Col 1 — before Platt scaling
      Col 2 — after Platt scaling
      Col 3 — reliability diagram (after)
    All on one figure for the poster.
    """
    n_models = len(models_data)
    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(11, 3.8 * n_models),
        squeeze=False
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.32,
                        left=0.09, right=0.97,
                        top=0.93, bottom=0.06)

    for row, m in enumerate(models_data):
        label  = m["label"]
        color  = m["color"]
        marker = m["marker"]

        for col, (tag, yp, bdf, ece_val) in enumerate([
            ("Before Platt scaling",
             m["y_pred_raw"], m["bin_df_raw"], m["ece_raw"]),
            ("After Platt scaling",
             m["y_pred_cal"], m["bin_df_cal"], m["ece_cal"]),
        ]):
            ax = axes[row][col]
            ax.plot([0,1],[0,1], color="#B4B2A9", lw=1.2,
                    linestyle="--", label="Perfect calibration")

            # Smooth isotonic curve
            x_iso, y_iso = iso_curve(m["y_true"], yp)
            ax.plot(x_iso, y_iso, color=color, lw=2,
                    label=f"ECE = {ece_val:.3f}")

            # Binned points
            bdf_v = bdf.dropna(subset=["mean_pred","frac_pos"])
            ax.scatter(bdf_v["mean_pred"], bdf_v["frac_pos"],
                       color=color, marker=marker,
                       s=bdf_v["count"]*0.6+15,
                       alpha=0.85, zorder=3)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Mean predicted probability", fontsize=9)
            ax.set_ylabel("Observed frequency", fontsize=9)
            title = f"{label}\n{tag}"
            ax.set_title(title, fontsize=9.5)
            ax.legend(fontsize=8, loc="upper left",
                      framealpha=0.9, edgecolor="#D3D1C7")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def plot_poster_figure(models_data, outpath):
    """
    Clean single-panel figure for the poster:
    After-Platt calibration curves for all models on one axes.
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0,1],[0,1], color="#B4B2A9", lw=1.2,
            linestyle="--", label="Perfect calibration", zorder=1)

    for m in models_data:
        yp  = m["y_pred_cal"]
        x, y = iso_curve(m["y_true"], yp)
        ax.plot(x, y, color=m["color"], lw=2.2,
                label=f"{m['label']}  (ECE={m['ece_cal']:.3f})")

        bdf_v = m["bin_df_cal"].dropna(subset=["mean_pred","frac_pos"])
        ax.scatter(bdf_v["mean_pred"], bdf_v["frac_pos"],
                   color=m["color"], marker=m["marker"],
                   s=bdf_v["count"]*0.6+15, alpha=0.75, zorder=3)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Observed frequency (mastectomy)", fontsize=11)
    ax.set_title("Calibration curves — after Platt scaling",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9, loc="upper left",
              framealpha=0.95, edgecolor="#D3D1C7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    print("=" * 60)
    print("Calibration Analysis with Platt Scaling")
    print("=" * 60)

    models_data  = []
    metrics_rows = []

    for m in MODELS:
        if not m["file"].exists():
            print(f"\n  Skipping {m['label']} — not found")
            continue

        print(f"\n  {m['label']}")
        df     = pd.read_csv(m["file"])
        y_true = df["y_true"].values.astype(int)
        y_raw  = df["y_pred"].values.astype(float)

        # Before Platt
        bdf_raw  = calibration_bins(y_true, y_raw, N_BINS)
        ece_raw  = ece_score(bdf_raw, len(y_true))
        brier_raw = brier_score_loss(y_true, y_raw)
        auc_raw   = roc_auc_score(y_true, y_raw)

        # Apply Platt scaling
        y_cal = platt_scale(y_true, y_raw)

        # After Platt
        bdf_cal  = calibration_bins(y_true, y_cal, N_BINS)
        ece_cal  = ece_score(bdf_cal, len(y_true))
        brier_cal = brier_score_loss(y_true, y_cal)
        auc_cal   = roc_auc_score(y_true, y_cal)

        print(f"    Before Platt: ECE={ece_raw:.4f}  "
              f"Brier={brier_raw:.4f}  AUC={auc_raw:.4f}")
        print(f"    After  Platt: ECE={ece_cal:.4f}  "
              f"Brier={brier_cal:.4f}  AUC={auc_cal:.4f}")
        improvement = ece_raw - ece_cal
        print(f"    ECE improvement: {improvement:+.4f}  "
              f"({'better' if improvement>0 else 'worse'})")

        models_data.append({
            "label":       m["label"],
            "color":       m["color"],
            "marker":      m["marker"],
            "y_true":      y_true,
            "y_pred_raw":  y_raw,
            "y_pred_cal":  y_cal,
            "bin_df_raw":  bdf_raw,
            "bin_df_cal":  bdf_cal,
            "ece_raw":     ece_raw,
            "ece_cal":     ece_cal,
        })

        metrics_rows.append({
            "model":      m["label"],
            "n":          len(y_true),
            "auc":        auc_raw,
            "ece_before": ece_raw,
            "ece_after":  ece_cal,
            "brier_before": brier_raw,
            "brier_after":  brier_cal,
            "ece_improved": ece_raw - ece_cal,
        })

        # Save calibrated OOF for downstream use
        tag = m["label"].replace(" ", "_").replace("—","").replace("(","").replace(")","")
        pd.DataFrame({
            "y_true":      y_true,
            "y_pred_raw":  y_raw,
            "y_pred_cal":  y_cal,
        }).to_csv(OUTPUT_DIR / f"oof_calibrated_{tag}.csv", index=False)

    if not models_data:
        print("\nNo OOF files found. Run surgery_pipeline.py A first.")
    else:
        # Before/after comparison figure
        plot_calibration_comparison(
            models_data,
            OUTPUT_DIR / "figure_calibration_comparison.png"
        )

        # Clean poster figure
        plot_poster_figure(
            models_data,
            OUTPUT_DIR / "figure_calibration_poster.png"
        )

        # Metrics table
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(
            OUTPUT_DIR / "calibration_metrics_platt.csv", index=False
        )

        print("\n" + "=" * 60)
        print("SUMMARY — ECE before vs after Platt scaling")
        print("=" * 60)
        print(f"\n{'Model':<35} {'ECE before':>10} {'ECE after':>10} {'Change':>8}")
        print("-" * 66)
        for r in metrics_rows:
            chg = r["ece_before"] - r["ece_after"]
            print(f"{r['model']:<35} {r['ece_before']:>10.4f} "
                  f"{r['ece_after']:>10.4f} {chg:>+8.4f}")

        print()
        print("AUC is preserved by Platt scaling (ranking unchanged).")
        print("Use calibrated predictions for any probability-based reporting.")
        print()
        print("Figures saved:")
        print("  figure_calibration_comparison.png  (before/after per model)")
        print("  figure_calibration_poster.png      (clean poster figure)")
        print()
        print("Done. Outputs in:", OUTPUT_DIR)
