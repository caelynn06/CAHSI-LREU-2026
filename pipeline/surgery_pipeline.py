"""
Surgery Type Prediction Pipeline
=================================
Target: Mastectomy (1) vs Breast-Conserving Surgery / BCS (0)

Cohort design
-------------
Option A (primary)   — no-neoadjuvant patients only (n≈557)
  MRI -> surgery directly, no intervening treatment.
  Clean causal path: pre-op MRI predicts surgical planning decision.

Option B (secondary) — neoadjuvant patients only (n≈303)
  MRI acquired pre-chemo, surgery post-chemo.
  Confounded by treatment response — reported as secondary analysis.

Both cohorts run through identical nested CV pipelines so results
are directly comparable. Option B is interpreted as:
"Does pre-treatment MRI predict the eventual surgery type,
accounting for treatment response?"

Key difference from T-stage pipeline
--------------------------------------
- TumorMajorAxisLength_mm, Volume_cu_mm_Tumor, SER_Total_tumor_vol_cu_mm
  are NOW LEGITIMATE predictors (tumor size drives the mastectomy decision)
- T stage is included as a clinical feature (known pre-op, used in planning)
- Feature set is larger and more interpretable

Three models per cohort
------------------------
  Imaging only    — full 530 radiomic features
  Clinical complete — 13 features (same 12 as before + T stage)
  Fusion          — imaging + clinical combined

Outputs (per cohort, suffix _A or _B)
---------------------------------------
  pipeline_outputs/oof_surgery_{model}_{cohort}.csv
  pipeline_outputs/surgery_results_{cohort}.csv
  pipeline_outputs/surgery_comparison_full.csv
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

IMAGING_PATH  = "imaging_only_with_target.csv"
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
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────

# Imaging features to DROP for surgery target
# NOTE: size features (MajorAxis, Volume, SER_Total) are now ALLOWED
IMAGING_DROP = [
    "early_stage",   # old target — remove entirely
]

# Clinical features to DROP — post-surgical outcomes and treatment vars
CLINICAL_DROP = [
    # Surgery fields (the target itself and related)
    "SURGERY | Surgery",
    "SURGERY | Days to Surgery (from the date of diagnosis)",
    # Neoadjuvant — used only for cohort splitting, not a predictor
    "Neoadjuvant therapy | Received Neoadjuvant Therapy or Not",
    # Treatment variables
    "Radiation Therapy | Neoadjuvant Radiation Therapy",
    "Radiation Therapy | Adjuvant Radiation Therapy",
    "Chemotherapy | Neoadjuvant Chemotherapy",
    "Chemotherapy | Adjuvant Chemotherapy",
    "Endocrine Therapy | Neoadjuvant Endocrine Therapy Medications",
    "Endocrine Therapy | Adjuvant Endocrine Therapy Medications",
    "Anti-Her2 Neu Therapy | Neoadjuvant Anti-Her2 Neu Therapy",
    "Anti-Her2 Neu Therapy | Adjuvant Anti-Her2 Neu Therapy",
    # Response outcomes
    "Tumor Response | Clinical Response, Evaluated Through Imaging",
    "Tumor Response | Pathologic Response to Neoadjuvant Therapy",
    "Near Complete Response | Overall Near-complete Response: Stricter Definition",
    "Near Complete Response | Overall Near-complete Response: Looser Definition",
    "Near Complete Response | Near-complete Response (Graded Measure)",
    "Pathologic Response to Neoadjuvant Therapy | Pathologic response to Neoadjuvant therapy: Pathologic stage (T) following neoadjuvant therapy",
    "Pathologic Response to Neoadjuvant Therapy | Pathologic response to Neoadjuvant therapy: Pathologic stage (N) following neoadjuvant therapy",
    "Pathologic Response to Neoadjuvant Therapy | Pathologic response to Neoadjuvant therapy: Pathologic stage (M) following neoadjuvant therapy",
    # Recurrence and survival
    "Recurrence | Recurrence event(s)",
    "Recurrence | Days to local recurrence (from the date of diagnosis)",
    "Recurrence | Days to distant recurrence(from the date of diagnosis)",
    "Follow Up | Days to death (from the date of diagnosis)",
    "Follow Up | Days to last local recurrence free assessment (from the date of diagnosis)",
    "Follow Up | Days to last distant recurrence free assemssment(from the date of diagnosis)",
    "Follow Up | Age at last contact in EMR f/u(days)(from the date of diagnosis) ,last time patient known to be alive, unless age of death is reported(in such case the age of death",
    # Scanner metadata
    "MRI Technical Information | Manufacturer",
    "MRI Technical Information | Manufacturer Model Name",
    "MRI Technical Information | Scan Options",
    "MRI Technical Information | Field Strength (Tesla)",
    "MRI Technical Information | Patient Position During MRI",
    "MRI Technical Information | Image Position of Patient",
    "MRI Technical Information | Contrast Agent",
    "MRI Technical Information | Contrast Bolus Volume (mL)",
    "MRI Technical Information | TR (Repetition Time)",
    "MRI Technical Information | TE (Echo Time)",
    "MRI Technical Information | Acquisition Matrix",
    "MRI Technical Information | Slice Thickness",
    "MRI Technical Information | Rows",
    "MRI Technical Information | Columns",
    "MRI Technical Information | Reconstruction Diameter",
    "MRI Technical Information | Flip Angle",
    "MRI Technical Information | FOV Computed (Field of View) in cm",
    "MRI Technical Information | Days to MRI (From the Date of Diagnosis)",
    # High-missingness columns (>80% missing, near-zero signal)
    "BIRADS DATA",
    "Mammography Characteristics | Mass Density",
    "Mammography Characteristics | Architectural distortion",
    "Mammography Characteristics | Age at mammo (days)",
    "Mammography Characteristics | Breast Density",
    "Mammography Characteristics | Shape",
    "Mammography Characteristics | Margin",
    "Mammography Characteristics | Tumor Size (cm)",
    "Mammography Characteristics | Calcifications",
    "US features | Shape",
    "US features | Margin",
    "US features | Tumor Size (cm)",
    "US features | Echogenicity",
    "US features | Solid",
    "US features | Posterior acoustic shadowing",
    "Tumor Characteristics | Oncotype score",
    "Endocrine Therapy | Number of Ovaries In Situ",
    "Tumor Characteristics | For Other Side If Bilateral",
    # Old target
    "early_stage",
    # The target itself
    "SURGERY | Definitive Surgery Type",
]

# Clean clinical features for clinical-complete model
# Now includes T stage (legitimate with surgery target)
CLINICAL_COMPLETE_FEATURES = [
    # Staging — known pre-op, directly used in surgical planning
    "Tumor Characteristics | Staging(Tumor Size)# [T]",
    "Tumor Characteristics | Staging(Nodes)#(Nx replaced by -1)[N]",
    # Receptor / molecular (from biopsy)
    "Tumor Characteristics | ER",
    "Tumor Characteristics | PR",
    "Tumor Characteristics | HER2",
    "Tumor Characteristics | Mol Subtype",
    # Grade
    "Tumor Characteristics | Tumor Grade",
    # Demographics
    "Demographics | Menopause (at diagnosis)",
    "Demographics | Race and Ethnicity",
    "Demographics | Date of Birth (Days)",
    # MRI clinical findings (radiologist-reported)
    "MRI Findings | Skin/Nipple Invovlement",
    "MRI Findings | Multicentric/Multifocal",
    "MRI Findings | Pec/Chest Involvement",
    "MRI Findings | Contralateral Breast Involvement",
    # Lymphadenopathy — now legitimate (not part of surgery target definition)
    "MRI Findings | Lymphadenopathy or Suspicious Nodes",
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


def bootstrap_auc_ci(y_true, y_pred, n_boot=1000, seed=42):
    rng  = np.random.default_rng(seed)
    aucs = []
    n    = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def delong_variance(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.0
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    v10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
    v01 = np.array([np.mean(p < pos) + 0.5 * np.mean(p == pos) for p in neg])
    return np.var(v10, ddof=1) / n1 + np.var(v01, ddof=1) / n0


def delong_test(y_true, pred_a, pred_b, label_a, label_b):
    auc_a = roc_auc_score(y_true, pred_a)
    auc_b = roc_auc_score(y_true, pred_b)
    se    = np.sqrt(delong_variance(y_true, pred_a) +
                    delong_variance(y_true, pred_b))
    z     = (auc_a - auc_b) / (se + 1e-12)
    p     = 2 * stats.norm.sf(abs(z))
    print(f"    {label_a} ({auc_a:.4f}) vs {label_b} ({auc_b:.4f}): "
          f"ΔAUC={auc_a-auc_b:+.4f}  z={z:.3f}  p={p:.4f}")
    return z, p


# ─────────────────────────────────────────────
# INNER LOOP — feature selection
# ─────────────────────────────────────────────

def select_features_inner(X_train, y_train):
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

    if len(selected) < 20:
        top_idx  = np.argsort(mean_imp)[::-1][:20]
        selected = [feat_names[i] for i in top_idx]

    return selected


# ─────────────────────────────────────────────
# NESTED CV
# ─────────────────────────────────────────────

def run_nested_cv(X, y, label, cohort_tag):
    cat_cols  = get_cat_cols(X)
    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)

    fold_aucs   = []
    fold_prs    = []
    fold_briers = []
    oof_preds   = np.zeros(len(y))
    oof_true    = np.zeros(len(y))

    for fold, (tr_idx, te_idx) in enumerate(
            outer_skf.split(X, y), start=1):

        print(f"  [{label}|{cohort_tag}] Fold {fold}/{OUTER_FOLDS}",
              end=" ... ")

        X_train = X.iloc[tr_idx]
        X_test  = X.iloc[te_idx]
        y_train = y.iloc[tr_idx]
        y_test  = y.iloc[te_idx]

        selected = select_features_inner(X_train, y_train)

        cat_sel = [c for c in cat_cols if c in selected]
        X_tr    = prep_for_catboost(X_train[selected], cat_sel)
        X_te    = prep_for_catboost(X_test[selected],  cat_sel)
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_sel]

        model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
        model.fit(X_tr, y_train, cat_features=cat_idx)

        proba = model.predict_proba(X_te)[:, 1]
        oof_preds[te_idx] = proba
        oof_true[te_idx]  = y_test.values

        auc = roc_auc_score(y_test, proba)
        fold_aucs.append(auc)
        fold_prs.append(average_precision_score(y_test, proba))
        fold_briers.append(brier_score_loss(y_test, proba))

        print(f"AUC={auc:.4f}  n_feat={len(selected)}")

    oof_auc    = roc_auc_score(oof_true, oof_preds)
    ci_lo, ci_hi = bootstrap_auc_ci(oof_true, oof_preds)

    print(f"\n  [{label}|{cohort_tag}] "
          f"OOF AUC={oof_auc:.4f}  CI=[{ci_lo:.4f},{ci_hi:.4f}]"
          f"  PR={np.mean(fold_prs):.4f}"
          f"  Brier={np.mean(fold_briers):.4f}")

    # Save OOF predictions
    tag = f"{label.replace(' ','_')}_{cohort_tag}"
    oof_df = pd.DataFrame({"y_true": oof_true.astype(int),
                           "y_pred": oof_preds})
    oof_df.to_csv(OUTPUT_DIR / f"oof_surgery_{tag}.csv", index=False)

    return {
        "label":       label,
        "cohort":      cohort_tag,
        "oof_auc":     oof_auc,
        "ci_lo":       ci_lo,
        "ci_hi":       ci_hi,
        "fold_aucs":   fold_aucs,
        "fold_prs":    fold_prs,
        "fold_briers": fold_briers,
        "oof_preds":   oof_preds,
        "oof_true":    oof_true,
    }


# ─────────────────────────────────────────────
# RUN ONE COHORT (imaging + clinical + fusion)
# ─────────────────────────────────────────────

def run_cohort(img_df, clin_df, y, cohort_tag):
    print(f"\n{'='*60}")
    print(f"COHORT: {cohort_tag}  (n={len(y)}, "
          f"mastectomy={y.sum()}, BCS={len(y)-y.sum()})")
    print(f"{'='*60}")

    # ── Build feature sets ────────────────────

    # Imaging
    drop_img = [c for c in img_df.columns if c in IMAGING_DROP]
    X_img    = img_df.drop(columns=drop_img, errors="ignore")

    # Clinical — drop leakage, missingness flags, high-missing cols
    drop_clin  = [c for c in clin_df.columns if c in CLINICAL_DROP]
    drop_clin += [c for c in clin_df.columns
                  if any(c.endswith(s)
                         for s in ["__is_NC","__is_NP","__is_NA"])]
    drop_clin += [c for c in clin_df.columns if c.startswith("has_")]
    X_clin_full = clin_df.drop(columns=drop_clin, errors="ignore")

    # Clinical complete — only the 15 hand-curated features
    avail_complete = [f for f in CLINICAL_COMPLETE_FEATURES
                      if f in clin_df.columns]
    X_clin_complete = clin_df[avail_complete].copy()

    # Fusion — imaging + clinical complete
    X_fusion = pd.concat([X_img, X_clin_complete], axis=1)

    print(f"\n  Feature counts:")
    print(f"    Imaging          : {X_img.shape[1]}")
    print(f"    Clinical complete: {X_clin_complete.shape[1]}")
    print(f"    Fusion           : {X_fusion.shape[1]}")

    results = []

    # ── Imaging only ──────────────────────────
    print(f"\n--- Imaging only ---")
    r = run_nested_cv(X_img, y, "Imaging", cohort_tag)
    results.append(r)

    # ── Clinical complete ─────────────────────
    print(f"\n--- Clinical complete ---")
    r = run_nested_cv(X_clin_complete, y, "Clinical", cohort_tag)
    results.append(r)

    # ── Fusion ────────────────────────────────
    print(f"\n--- Fusion ---")
    r = run_nested_cv(X_fusion, y, "Fusion", cohort_tag)
    results.append(r)

    # ── DeLong comparisons ────────────────────
    print(f"\n--- DeLong comparisons ({cohort_tag}) ---")
    img_r  = results[0]
    clin_r = results[1]
    fus_r  = results[2]
    delong_test(img_r["oof_true"],  img_r["oof_preds"],
                clin_r["oof_preds"], "Imaging", "Clinical")
    delong_test(fus_r["oof_true"],  fus_r["oof_preds"],
                img_r["oof_preds"],  "Fusion",  "Imaging")
    delong_test(fus_r["oof_true"],  fus_r["oof_preds"],
                clin_r["oof_preds"], "Fusion",  "Clinical")

    # ── Save summary ──────────────────────────
    rows = []
    for r in results:
        rows.append({
            "cohort":          cohort_tag,
            "model":           r["label"],
            "n":               len(y),
            "n_mastectomy":    int(y.sum()),
            "oof_auc":         r["oof_auc"],
            "ci_lo":           r["ci_lo"],
            "ci_hi":           r["ci_hi"],
            "fold_auc_mean":   np.mean(r["fold_aucs"]),
            "fold_auc_std":    np.std(r["fold_aucs"]),
            "fold_prauc_mean": np.mean(r["fold_prs"]),
            "fold_brier_mean": np.mean(r["fold_briers"]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / f"surgery_results_{cohort_tag}.csv",
                   index=False)
    print(f"\n  Saved: surgery_results_{cohort_tag}.csv")

    return results


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_and_split():
    img  = pd.read_csv(IMAGING_PATH).set_index("Patient ID")
    clin = pd.read_csv(CLINICAL_PATH).set_index("Patient ID")

    # Align
    shared = img.index.intersection(clin.index)
    img    = img.loc[shared]
    clin   = clin.loc[shared]

    target_col = "SURGERY | Definitive Surgery Type"
    neoadj_col = "Neoadjuvant therapy | Received Neoadjuvant Therapy or Not"

    # Drop patients with missing surgery type
    valid = clin[target_col].notna()
    clin  = clin[valid]
    img   = img[img.index.isin(clin.index)]

    y_all = clin[target_col].astype(int)

    # Cohort A — no neoadjuvant (neoadj==2)
    mask_A = clin[neoadj_col] == 2
    clin_A = clin[mask_A]
    img_A  = img[img.index.isin(clin_A.index)]
    y_A    = y_all[mask_A]

    # Cohort B — neoadjuvant (neoadj==1)
    mask_B = clin[neoadj_col] == 1
    clin_B = clin[mask_B]
    img_B  = img[img.index.isin(clin_B.index)]
    y_B    = y_all[mask_B]

    print("Data loaded:")
    print(f"  Cohort A (no neoadjuvant): n={len(y_A)} "
          f"[BCS={int((y_A==0).sum())}  Mastectomy={int(y_A.sum())}]")
    print(f"  Cohort B (neoadjuvant)   : n={len(y_B)} "
          f"[BCS={int((y_B==0).sum())}  Mastectomy={int(y_B.sum())}]")

    return (img_A, clin_A, y_A), (img_B, clin_B, y_B)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Accept optional argument: 'A', 'B', or 'both' (default)
    run_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "BOTH"

    print("=" * 60)
    print("Surgery Type Prediction Pipeline")
    print(f"Running: Cohort {run_arg}")
    print("=" * 60)

    cohort_A, cohort_B = load_and_split()

    all_results = []

    if run_arg in ("A", "BOTH"):
        img_A, clin_A, y_A = cohort_A
        results_A = run_cohort(img_A, clin_A, y_A, "A_no_neoadj")
        all_results.extend(results_A)

    if run_arg in ("B", "BOTH"):
        img_B, clin_B, y_B = cohort_B
        results_B = run_cohort(img_B, clin_B, y_B, "B_neoadj")
        all_results.extend(results_B)

    # Combined comparison table
    if run_arg == "BOTH" and all_results:
        rows = []
        for r in all_results:
            rows.append({
                "cohort":          r["cohort"],
                "model":           r["label"],
                "oof_auc":         r["oof_auc"],
                "ci_lo":           r["ci_lo"],
                "ci_hi":           r["ci_hi"],
                "fold_auc_mean":   np.mean(r["fold_aucs"]),
                "fold_auc_std":    np.std(r["fold_aucs"]),
                "fold_prauc_mean": np.mean(r["fold_prs"]),
                "fold_brier_mean": np.mean(r["fold_briers"]),
            })
        full_df = pd.DataFrame(rows)
        full_df.to_csv(OUTPUT_DIR / "surgery_comparison_full.csv",
                       index=False)
        print("\nFull comparison saved: surgery_comparison_full.csv")
        print(full_df.to_string(index=False))

    print("\nDone. Outputs in:", OUTPUT_DIR)
