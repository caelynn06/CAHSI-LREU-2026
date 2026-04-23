"""
Clinical-Complete Model
=======================
Runs the same nested CV pipeline as staging_pipeline_clean.py
but using ONLY the 12 legitimate clinical features that have
<5% missing values and no leakage concerns.

This is the honest clinical baseline — what a clinician knows
from biopsy results + physical/MRI exam alone, before any
radiomic analysis.

Outputs
-------
  pipeline_outputs/oof_Clinical_complete.csv
  pipeline_outputs/clinical_complete_summary.csv
  pipeline_outputs/clinical_complete_vs_others.csv  (comparison table)

Run this AFTER staging_pipeline_clean.py so the imaging and
clinical-full OOF files already exist for comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier
from scipy import stats

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CLINICAL_PATH = "clean_clinical_rebuilt.csv"
OUTPUT_DIR    = Path("pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTER_FOLDS  = 5
INNER_FOLDS  = 5
RANDOM_STATE = 42
IMPORTANCE_THRESHOLD = 0.001

BASE_CATBOOST_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=False,
    random_seed=RANDOM_STATE,
)

# ─────────────────────────────────────────────
# THE 12 CLINICAL-COMPLETE FEATURES
#
# Selection criteria:
#   - < 5% missing values
#   - Available at time of diagnosis (no treatment/outcome leakage)
#   - Not a proxy for tumor size (which defines the target)
#   - Biologically or clinically meaningful for breast cancer
#
# Deliberately excluded despite low missingness:
#   - Tumor Location (L/R laterality — no T-stage relationship)
#   - Position (259 unique free-text values, not encodable)
#   - Bilateral Information (corr=+0.004, near-zero signal)
#   - Nottingham grade (31% missing, also overlaps with Tumor Grade)
#   - Contralateral involvement (corr=−0.020, near-zero)
# ─────────────────────────────────────────────

CLINICAL_COMPLETE_FEATURES = [
    # Receptor / molecular status (from core biopsy)
    "Tumor Characteristics | ER",
    "Tumor Characteristics | PR",
    "Tumor Characteristics | HER2",
    "Tumor Characteristics | Mol Subtype",

    # Pathologic grade (from core biopsy)
    "Tumor Characteristics | Tumor Grade",       # 1.6% missing

    # Patient demographics
    "Demographics | Menopause (at diagnosis)",
    "Demographics | Race and Ethnicity",
    "Demographics | Date of Birth (Days)",       # age proxy

    # MRI clinical findings (radiologist-reported, not radiomic)
    "MRI Findings | Skin/Nipple Invovlement",
    "MRI Findings | Multicentric/Multifocal",
    "MRI Findings | Pec/Chest Involvement",
    "MRI Findings | Contralateral Breast Involvement",
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_cat_cols(df):
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def prep_for_catboost(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].fillna("Missing").astype(str)
    for c in [x for x in df.columns if x not in cat_cols]:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def delong_auc_variance(y_true, y_pred):
    """DeLong variance estimate for a single model's AUC."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.0
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    v10 = np.array([
        np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos
    ])
    v01 = np.array([
        np.mean(p < pos) + 0.5 * np.mean(p == pos) for p in neg
    ])
    return np.var(v10, ddof=1) / n1 + np.var(v01, ddof=1) / n0


def delong_test(y_true, pred_a, pred_b, label_a="A", label_b="B"):
    """Paired DeLong test comparing two models on same OOF predictions."""
    auc_a  = roc_auc_score(y_true, pred_a)
    auc_b  = roc_auc_score(y_true, pred_b)
    var_a  = delong_auc_variance(y_true, pred_a)
    var_b  = delong_auc_variance(y_true, pred_b)
    se     = np.sqrt(var_a + var_b)
    z      = (auc_a - auc_b) / (se + 1e-12)
    p      = 2 * stats.norm.sf(abs(z))
    print(f"  {label_a} AUC={auc_a:.4f}  vs  {label_b} AUC={auc_b:.4f}")
    print(f"  ΔAUC={auc_a-auc_b:+.4f}  z={z:.3f}  p={p:.4f}")
    return auc_a, auc_b, z, p


def bootstrap_auc_ci(y_true, y_pred, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap 95% CI on OOF AUC."""
    rng   = np.random.default_rng(seed)
    aucs  = []
    n     = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    lo = np.percentile(aucs, 100 * (1 - ci) / 2)
    hi = np.percentile(aucs, 100 * (1 + ci) / 2)
    return lo, hi


# ─────────────────────────────────────────────
# INNER LOOP — feature selection
# ─────────────────────────────────────────────

def select_features_inner(X_train, y_train):
    """
    With only 12 features, selection is less critical —
    but we keep the same structure for consistency with
    the main pipeline.
    """
    cat_cols   = get_cat_cols(X_train)
    X_prep     = prep_for_catboost(X_train, cat_cols)
    cat_idx    = [X_prep.columns.get_loc(c) for c in cat_cols]
    feat_names = X_prep.columns.tolist()
    all_imp    = []

    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    for tr, va in skf.split(X_prep, y_train):
        model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
        model.fit(X_prep.iloc[tr], y_train.iloc[tr],
                  cat_features=cat_idx)
        res = permutation_importance(
            model, X_prep.iloc[va], y_train.iloc[va],
            n_repeats=3, random_state=RANDOM_STATE,
            scoring="roc_auc"
        )
        all_imp.append(res.importances_mean)

    mean_imp = np.array(all_imp).mean(axis=0)
    selected = [f for f, v in zip(feat_names, mean_imp)
                if v > IMPORTANCE_THRESHOLD]

    # With only 12 features, always keep at least 6
    if len(selected) < 6:
        top_idx  = np.argsort(mean_imp)[::-1][:6]
        selected = [feat_names[i] for i in top_idx]

    return selected, dict(zip(feat_names, mean_imp))


# ─────────────────────────────────────────────
# NESTED CV
# ─────────────────────────────────────────────

def run_nested_cv(X, y, label="Clinical complete"):
    cat_cols  = get_cat_cols(X)
    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)

    fold_aucs   = []
    fold_prs    = []
    fold_briers = []
    fold_nfeats = []
    oof_preds   = np.zeros(len(y))
    oof_true    = np.zeros(len(y))

    # Track importance across folds
    all_feat_names = X.columns.tolist()
    importance_matrix = pd.DataFrame(
        0.0,
        index=all_feat_names,
        columns=range(1, OUTER_FOLDS + 1)
    )

    for fold, (tr_idx, te_idx) in enumerate(
            outer_skf.split(X, y), start=1):

        print(f"  [{label}] Fold {fold}/{OUTER_FOLDS} ...", end=" ")

        X_train = X.iloc[tr_idx]
        X_test  = X.iloc[te_idx]
        y_train = y.iloc[tr_idx]
        y_test  = y.iloc[te_idx]

        selected, imp_scores = select_features_inner(X_train, y_train)

        for feat, score in imp_scores.items():
            if feat in importance_matrix.index:
                importance_matrix.loc[feat, fold] = max(score, 0)

        fold_nfeats.append(len(selected))

        cat_sel   = [c for c in cat_cols if c in selected]
        X_tr      = prep_for_catboost(X_train[selected], cat_sel)
        X_te      = prep_for_catboost(X_test[selected],  cat_sel)
        cat_idx   = [X_tr.columns.get_loc(c) for c in cat_sel]

        model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
        model.fit(X_tr, y_train, cat_features=cat_idx)

        proba = model.predict_proba(X_te)[:, 1]
        oof_preds[te_idx] = proba
        oof_true[te_idx]  = y_test.values

        auc    = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        brier  = brier_score_loss(y_test, proba)

        fold_aucs.append(auc)
        fold_prs.append(pr_auc)
        fold_briers.append(brier)

        print(f"AUC={auc:.4f}  n_features={len(selected)}")

    oof_auc     = roc_auc_score(oof_true, oof_preds)
    ci_lo, ci_hi = bootstrap_auc_ci(oof_true, oof_preds)

    print(f"\n  [{label}] Summary:")
    print(f"    OOF AUC (pooled)    : {oof_auc:.4f}")
    print(f"    95% CI (bootstrap)  : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Mean fold AUC       : {np.mean(fold_aucs):.4f} "
          f"± {np.std(fold_aucs):.4f}")
    print(f"    Mean fold PR-AUC    : {np.mean(fold_prs):.4f} "
          f"± {np.std(fold_prs):.4f}")
    print(f"    Mean Brier score    : {np.mean(fold_briers):.4f} "
          f"± {np.std(fold_briers):.4f}")
    print(f"    Features per fold   : {np.mean(fold_nfeats):.1f} "
          f"± {np.std(fold_nfeats):.1f}")

    return {
        "label":        label,
        "oof_auc":      oof_auc,
        "ci_lo":        ci_lo,
        "ci_hi":        ci_hi,
        "fold_aucs":    fold_aucs,
        "fold_prs":     fold_prs,
        "fold_briers":  fold_briers,
        "fold_nfeats":  fold_nfeats,
        "oof_preds":    oof_preds,
        "oof_true":     oof_true,
        "importance":   importance_matrix,
    }


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE SUMMARY
# ─────────────────────────────────────────────

def summarize_importance(result):
    imp = result["importance"]
    df  = pd.DataFrame({
        "feature":         imp.index,
        "mean_importance": imp.mean(axis=1).values,
        "std_importance":  imp.std(axis=1).values,
        "fold_count":      (imp > IMPORTANCE_THRESHOLD).sum(axis=1).values,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ─────────────────────────────────────────────
# COMPARE TO OTHER MODELS
# ─────────────────────────────────────────────

def compare_all_models(result_complete):
    """
    Load OOF files from the main pipeline and compare all models.
    Adds bootstrap CIs to all models for fair comparison.
    """
    oof_files = {
        "Imaging only":    OUTPUT_DIR / "oof_Imaging_only.csv",
        "Clinical full":   OUTPUT_DIR / "oof_Clinical_only.csv",
        "Fusion":          OUTPUT_DIR / "oof_Fusion.csv",
    }

    results = {}
    for label, path in oof_files.items():
        if not path.exists():
            print(f"  Warning: {path} not found — skipping {label}")
            continue
        df   = pd.read_csv(path)
        yt   = df["y_true"].values.astype(int)
        yp   = df["y_pred"].values.astype(float)
        auc  = roc_auc_score(yt, yp)
        lo, hi = bootstrap_auc_ci(yt, yp)
        results[label] = {
            "oof_auc": auc, "ci_lo": lo, "ci_hi": hi,
            "oof_true": yt, "oof_preds": yp,
            "fold_briers": [brier_score_loss(yt, yp)],
            "fold_prs":    [average_precision_score(yt, yp)],
        }

    # Add clinical-complete
    yt = result_complete["oof_true"].astype(int)
    yp = result_complete["oof_preds"]
    results["Clinical complete"] = {
        "oof_auc":  result_complete["oof_auc"],
        "ci_lo":    result_complete["ci_lo"],
        "ci_hi":    result_complete["ci_hi"],
        "oof_true": yt, "oof_preds": yp,
        "fold_briers": result_complete["fold_briers"],
        "fold_prs":    result_complete["fold_prs"],
    }

    print("\n" + "=" * 60)
    print("FULL MODEL COMPARISON")
    print("=" * 60)

    rows = []
    order = ["Imaging only", "Fusion",
             "Clinical full", "Clinical complete"]

    for label in order:
        if label not in results:
            continue
        r   = results[label]
        auc = r["oof_auc"]
        lo  = r["ci_lo"]
        hi  = r["ci_hi"]
        br  = np.mean(r["fold_briers"])
        pr  = np.mean(r["fold_prs"])
        print(f"\n  {label}")
        print(f"    AUC  : {auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"    PR   : {pr:.4f}")
        print(f"    Brier: {br:.4f}")
        rows.append({
            "model":    label,
            "oof_auc":  auc,
            "ci_lo":    lo,
            "ci_hi":    hi,
            "pr_auc":   pr,
            "brier":    br,
        })

    # DeLong pairwise vs imaging
    print("\n" + "=" * 60)
    print("DeLong comparisons vs imaging only")
    print("=" * 60)
    if "Imaging only" in results:
        ref = results["Imaging only"]
        for label in ["Clinical full", "Clinical complete", "Fusion"]:
            if label not in results:
                continue
            print(f"\n  Imaging only vs {label}:")
            delong_test(
                ref["oof_true"],
                ref["oof_preds"],
                results[label]["oof_preds"],
                "Imaging only", label
            )

    # Clinical full vs clinical complete
    if "Clinical full" in results and "Clinical complete" in results:
        print("\n  Clinical full vs Clinical complete:")
        delong_test(
            results["Clinical full"]["oof_true"],
            results["Clinical full"]["oof_preds"],
            results["Clinical complete"]["oof_preds"],
            "Clinical full", "Clinical complete"
        )

    # Save comparison table
    comp_df = pd.DataFrame(rows)
    comp_path = OUTPUT_DIR / "clinical_complete_vs_others.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\nComparison table saved: {comp_path}")
    return comp_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Clinical-Complete Model")
    print("=" * 60)

    # ── Load data ─────────────────────────────
    clin = pd.read_csv(CLINICAL_PATH)
    clin = clin.set_index("Patient ID")

    valid = clin["early_stage"].notna()
    clin  = clin[valid]
    y     = clin["early_stage"].astype(int)

    # ── Verify all features exist ──────────────
    missing_cols = [f for f in CLINICAL_COMPLETE_FEATURES
                    if f not in clin.columns]
    if missing_cols:
        print("WARNING — these features not found in dataset:")
        for c in missing_cols:
            print(f"  {c}")

    available = [f for f in CLINICAL_COMPLETE_FEATURES
                 if f in clin.columns]
    X = clin[available].copy()

    print(f"\nPatients : {len(y)}")
    print(f"Features : {len(available)}")
    print(f"Class    : {y.value_counts().to_dict()}")
    print("\nFeature set:")
    for f in available:
        n_miss = X[f].isna().sum()
        pct    = 100 * n_miss / len(X)
        print(f"  {pct:4.1f}% missing | {f}")

    # ── Run nested CV ──────────────────────────
    print("\nRunning nested CV ...")
    result = run_nested_cv(X, y, label="Clinical complete")

    # ── Save OOF predictions ───────────────────
    oof_df = pd.DataFrame({
        "y_true": result["oof_true"].astype(int),
        "y_pred": result["oof_preds"],
    })
    oof_path = OUTPUT_DIR / "oof_Clinical_complete.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved: {oof_path}")

    # ── Feature importance ─────────────────────
    print("\n" + "=" * 60)
    print("Feature importance (clinical-complete model)")
    print("=" * 60)
    imp_df = summarize_importance(result)
    print(imp_df[["rank", "feature", "mean_importance",
                  "std_importance", "fold_count"]].to_string(index=False))

    imp_path = OUTPUT_DIR / "clinical_complete_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"\nImportance saved: {imp_path}")

    # ── Save per-fold summary ──────────────────
    summary = pd.DataFrame({
        "metric":  ["oof_auc", "ci_lo", "ci_hi",
                    "fold_auc_mean", "fold_auc_std",
                    "fold_prauc_mean", "fold_brier_mean"],
        "value":   [
            result["oof_auc"],
            result["ci_lo"],
            result["ci_hi"],
            np.mean(result["fold_aucs"]),
            np.std(result["fold_aucs"]),
            np.mean(result["fold_prs"]),
            np.mean(result["fold_briers"]),
        ]
    })
    summary_path = OUTPUT_DIR / "clinical_complete_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    # ── Full model comparison ──────────────────
    comp_df = compare_all_models(result)

    print("\n" + "=" * 60)
    print("Done. Files saved to:", OUTPUT_DIR)
    print("=" * 60)
