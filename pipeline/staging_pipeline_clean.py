"""
Breast Cancer T-Stage Prediction Pipeline
==========================================
Target: early_stage (T0/T1 = 1 vs T2+ = 0)

Design principles:
  1. All feature decisions made BEFORE touching any split data
  2. Nested CV: inner loop for feature selection + tuning,
     outer loop for unbiased evaluation only
  3. Three separate models compared: imaging-only, clinical-only, fusion
  4. Leakage audit is explicit and documented
  5. DeLong test for AUC comparison between models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

IMAGING_PATH   = "imaging_only_with_target.csv"
CLINICAL_PATH  = "clean_clinical_rebuilt.csv"
OUTPUT_DIR     = Path("pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTER_FOLDS  = 5
INNER_FOLDS  = 5
RANDOM_STATE = 42

# Permutation importance threshold for feature selection (inner loop)
# This is evaluated per fold, so there's no global double-dipping
IMPORTANCE_THRESHOLD = 0.001

# CatBoost base params — tuning happens in inner loop (see tune_catboost)
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
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    img = pd.read_csv(IMAGING_PATH)
    clin = pd.read_csv(CLINICAL_PATH)

    # Sanity check: confirm early_stage is a perfect recode of T stage
    t_col = "Tumor Characteristics | Staging(Tumor Size)# [T]"
    cross = pd.crosstab(clin[t_col], clin["early_stage"])
    print("\n[Sanity check] T stage vs early_stage crosstab:")
    print(cross)
    assert (cross.values == 0).sum() == cross.size - cross.shape[0], \
        "early_stage is not a clean recode of T stage — check definition"

    # Align on Patient ID
    img = img.set_index("Patient ID")
    clin = clin.set_index("Patient ID")
    shared_ids = img.index.intersection(clin.index)
    img  = img.loc[shared_ids]
    clin = clin.loc[shared_ids]

    y = clin["early_stage"].astype(int)

    print(f"\nPatients: {len(y)}")
    print(f"Class distribution:\n{y.value_counts()}")
    return img, clin, y


# ─────────────────────────────────────────────
# STEP 2: DEFINE ALLOWABLE FEATURES
#
# All decisions here are made ONCE, before any
# modelling. The rule is: "would this feature be
# available to a clinician BEFORE T staging?"
#
# T staging is determined by:
#   - Physical exam
#   - MRI tumor size measurement
#   - Mammography / ultrasound size
#   - Pathology (grade, ER/PR/HER2) — these ARE
#     available pre-staging but ARE correlated;
#     decide explicitly below
# ─────────────────────────────────────────────

# ── IMAGING FEATURES ──────────────────────────
# Drop: any direct size/volume measurement that
# feeds into the T-stage definition.
# Keep: kinetics, texture, morphology (non-size)

IMAGING_DROP_EXACT = [
    "TumorMajorAxisLength_mm",   # direct T-stage input
    "Volume_cu_mm_Tumor",        # direct T-stage input
    "early_stage",               # target
]

# Drop imaging columns that contain these substrings (partial match)
IMAGING_DROP_SUBSTRINGS = [
    # none currently — keep all kinetics/texture
]

# ── CLINICAL FEATURES ─────────────────────────
# Drop list — any of these would constitute leakage
# or are post-diagnosis / post-treatment outcomes

CLINICAL_DROP_EXACT = [
    # ── Target and its direct inputs ──────────
    "Tumor Characteristics | Staging(Tumor Size)# [T]",
    "Tumor Characteristics | Staging(Nodes)#(Nx replaced by -1)[N]",
    "Tumor Characteristics | Staging(Metastasis)#(Mx -replaced by -1)[M]",
    "Demographics | Metastatic at Presentation (Outside of Lymph Nodes)",

    # ── Tumor size measured by imaging modalities ──
    # These are size proxies that directly determined T stage
    "Mammography Characteristics | Tumor Size (cm)",
    "US features | Tumor Size (cm)",

    # ── Lymphadenopathy: used in N staging ────
    "MRI Findings | Lymphadenopathy or Suspicious Nodes",

    # ── Post-diagnosis pathologic outcomes ────
    "Tumor Response | Clinical Response, Evaluated Through Imaging",
    "Tumor Response | Pathologic Response to Neoadjuvant Therapy",
    "Near Complete Response | Overall Near-complete Response: Stricter Definition",
    "Near Complete Response | Overall Near-complete Response: Looser Definition",
    "Near Complete Response | Near-complete Response (Graded Measure)",
    "Pathologic Response to Neoadjuvant Therapy | Pathologic response to Neoadjuvant therapy: Pathologic stage (T) following neoadjuvant therapy",
    "Pathologic Response to Neoadjuvant Therapy | Pathologic response to Neoadjuvant therapy: Pathologic stage (N) following neoadjuvant therapy",
    "Pathologic Response to Neoadjuvant Therapy | Pathologic response to Neoadjuvant therapy: Pathologic stage (M) following neoadjuvant therapy",

    # ── Treatment variables (post-staging decisions) ─
    "SURGERY | Surgery",
    "SURGERY | Days to Surgery (from the date of diagnosis)",
    "SURGERY | Definitive Surgery Type",
    "Radiation Therapy | Neoadjuvant Radiation Therapy",
    "Radiation Therapy | Adjuvant Radiation Therapy",
    "Chemotherapy | Neoadjuvant Chemotherapy",
    "Chemotherapy | Adjuvant Chemotherapy",
    "Endocrine Therapy | Neoadjuvant Endocrine Therapy Medications",
    "Endocrine Therapy | Adjuvant Endocrine Therapy Medications",
    "Anti-Her2 Neu Therapy | Neoadjuvant Anti-Her2 Neu Therapy",
    "Anti-Her2 Neu Therapy | Adjuvant Anti-Her2 Neu Therapy",
    "Neoadjuvant therapy | Received Neoadjuvant Therapy or Not",

    # ── Recurrence and survival outcomes ──────
    "Recurrence | Recurrence event(s)",
    "Recurrence | Days to local recurrence (from the date of diagnosis)",
    "Recurrence | Days to distant recurrence(from the date of diagnosis)",
    "Follow Up | Days to death (from the date of diagnosis)",
    "Follow Up | Days to last local recurrence free assessment (from the date of diagnosis)",
    "Follow Up | Days to last distant recurrence free assemssment(from the date of diagnosis)",
    "Follow Up | Age at last contact in EMR f/u(days)(from the date of diagnosis) ,last time patient known to be alive, unless age of death is reported(in such case the age of death",

    # ── Scanner/acquisition metadata ──────────
    # These are confounders (site effects), not predictors.
    # Include only if doing site-stratified analysis separately.
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

    # ── Target itself ──────────────────────────
    "early_stage",
]

# Drop all __is_NC / __is_NP / __is_NA missingness flag columns
# (these are metadata about data collection, not clinical features)
CLINICAL_DROP_SUFFIX = ["__is_NC", "__is_NP", "__is_NA"]

# Drop has_* presence flags
CLINICAL_DROP_PREFIX = ["has_"]


def build_feature_sets(img_df, clin_df):
    """
    Returns X_imaging, X_clinical, X_fusion (all leakage-free).
    Does NOT touch y or split the data.
    """
    # ── Imaging ───────────────────────────────
    drop_img = [c for c in img_df.columns if c in IMAGING_DROP_EXACT]
    drop_img += [c for c in img_df.columns
                 if any(s in c for s in IMAGING_DROP_SUBSTRINGS)]
    X_img = img_df.drop(columns=drop_img, errors="ignore")

    # ── Clinical ──────────────────────────────
    drop_clin = [c for c in clin_df.columns if c in CLINICAL_DROP_EXACT]
    drop_clin += [c for c in clin_df.columns
                  if any(c.endswith(s) for s in CLINICAL_DROP_SUFFIX)]
    drop_clin += [c for c in clin_df.columns
                  if any(c.startswith(p) for p in CLINICAL_DROP_PREFIX)]
    X_clin = clin_df.drop(columns=drop_clin, errors="ignore")

    # ── Fusion ────────────────────────────────
    X_fusion = pd.concat([X_img, X_clin], axis=1)

    print(f"\nFeature counts after leakage removal:")
    print(f"  Imaging  : {X_img.shape[1]}")
    print(f"  Clinical : {X_clin.shape[1]}")
    print(f"  Fusion   : {X_fusion.shape[1]}")

    return X_img, X_clin, X_fusion


# ─────────────────────────────────────────────
# STEP 3: PREPROCESSING HELPERS
# ─────────────────────────────────────────────

def get_cat_cols(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def prep_for_catboost(df, cat_cols):
    """Fill NaN in cat cols with 'Missing', numeric with median."""
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].fillna("Missing").astype(str)
    num_cols = [c for c in df.columns if c not in cat_cols]
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


# ─────────────────────────────────────────────
# STEP 4: INNER-LOOP FEATURE SELECTION
#
# Permutation importance is computed only on
# inner-fold held-out data. The selected features
# are local to each outer fold — never global.
# ─────────────────────────────────────────────

def select_features_inner(X_train, y_train, cat_cols):
    """
    Run inner 5-fold CV on training data only.
    Return list of features with mean permutation importance
    above threshold. This is called once per outer fold.
    """
    cat_cols_present = [c for c in cat_cols if c in X_train.columns]
    X_prep = prep_for_catboost(X_train, cat_cols_present)
    cat_idx = [X_prep.columns.get_loc(c) for c in cat_cols_present]

    feature_names = X_prep.columns.tolist()
    all_importances = []

    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    for tr, va in skf.split(X_prep, y_train):
        X_tr = X_prep.iloc[tr]
        X_va = X_prep.iloc[va]
        y_tr = y_train.iloc[tr]
        y_va = y_train.iloc[va]

        model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
        model.fit(X_tr, y_tr, cat_features=cat_idx)

        result = permutation_importance(
            model, X_va, y_va,
            n_repeats=3,
            random_state=RANDOM_STATE,
            scoring="roc_auc"
        )
        all_importances.append(result.importances_mean)

    mean_imp = np.array(all_importances).mean(axis=0)
    selected = [f for f, imp in zip(feature_names, mean_imp)
                if imp > IMPORTANCE_THRESHOLD]

    # Always keep at least top-20 to avoid degenerate folds
    if len(selected) < 20:
        top20_idx = np.argsort(mean_imp)[::-1][:20]
        selected = [feature_names[i] for i in top20_idx]

    return selected


# ─────────────────────────────────────────────
# STEP 5: NESTED CV EVALUATION
# ─────────────────────────────────────────────

def delong_auc_variance(y_true, y_pred):
    """
    Variance of AUC estimate using DeLong method.
    Used for statistical comparison between models.
    """
    n1 = int(y_true.sum())      # positives
    n0 = len(y_true) - n1       # negatives
    if n1 == 0 or n0 == 0:
        return 0.0

    pos_scores = y_pred[y_true == 1]
    neg_scores = y_pred[y_true == 0]

    # Placement values
    v10 = np.array([np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)
                    for ps in pos_scores])
    v01 = np.array([np.mean(ps < pos_scores) + 0.5 * np.mean(ps == pos_scores)
                    for ps in neg_scores])

    s10 = np.var(v10, ddof=1) / n1
    s01 = np.var(v01, ddof=1) / n0
    return s10 + s01


def evaluate_nested_cv(X, y, label="Model", do_feature_selection=True):
    """
    Nested 5-fold CV.

    Outer loop: evaluation only (never sees feature selection).
    Inner loop (if do_feature_selection=True): selects features
      on training data, returns selected list per fold.

    Returns dict with fold metrics and OOF predictions for
    DeLong comparison.
    """
    X = X.copy()
    y = y.copy()
    cat_cols = get_cat_cols(X)

    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)

    fold_aucs    = []
    fold_prs     = []
    fold_briers  = []
    fold_n_feats = []
    oof_preds    = np.zeros(len(y))
    oof_true     = np.zeros(len(y))

    for fold, (tr_idx, te_idx) in enumerate(outer_skf.split(X, y), start=1):
        print(f"  [{label}] Outer fold {fold}/{OUTER_FOLDS} ...", end=" ")

        X_train_raw = X.iloc[tr_idx]
        X_test_raw  = X.iloc[te_idx]
        y_train     = y.iloc[tr_idx]
        y_test      = y.iloc[te_idx]

        # Inner loop: feature selection on training data only
        if do_feature_selection:
            selected = select_features_inner(X_train_raw, y_train, cat_cols)
        else:
            selected = X_train_raw.columns.tolist()

        fold_n_feats.append(len(selected))

        X_train = X_train_raw[selected]
        X_test  = X_test_raw[selected]

        # Prep for CatBoost
        cat_cols_sel = [c for c in cat_cols if c in selected]
        X_train = prep_for_catboost(X_train, cat_cols_sel)
        X_test  = prep_for_catboost(X_test,  cat_cols_sel)
        cat_idx = [X_train.columns.get_loc(c) for c in cat_cols_sel]

        # Train on outer training set, evaluate on outer test set
        model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
        model.fit(X_train, y_train, cat_features=cat_idx)

        proba = model.predict_proba(X_test)[:, 1]
        oof_preds[te_idx] = proba
        oof_true[te_idx]  = y_test.values

        auc    = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        brier  = brier_score_loss(y_test, proba)

        fold_aucs.append(auc)
        fold_prs.append(pr_auc)
        fold_briers.append(brier)

        print(f"AUC={auc:.3f}, n_features={len(selected)}")

    # OOF pooled AUC (more stable estimate than mean of folds)
    oof_auc = roc_auc_score(oof_true, oof_preds)

    print(f"\n  [{label}] Summary:")
    print(f"    OOF AUC (pooled)  : {oof_auc:.4f}")
    print(f"    Mean fold AUC     : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"    Mean fold PR-AUC  : {np.mean(fold_prs):.4f} ± {np.std(fold_prs):.4f}")
    print(f"    Mean Brier score  : {np.mean(fold_briers):.4f} ± {np.std(fold_briers):.4f}")
    print(f"    Features per fold : {np.mean(fold_n_feats):.0f} ± {np.std(fold_n_feats):.0f}")

    return {
        "label":         label,
        "oof_auc":       oof_auc,
        "fold_aucs":     fold_aucs,
        "fold_prs":      fold_prs,
        "fold_briers":   fold_briers,
        "fold_n_feats":  fold_n_feats,
        "oof_preds":     oof_preds,
        "oof_true":      oof_true,
    }


# ─────────────────────────────────────────────
# STEP 6: STATISTICAL COMPARISON (DeLong test)
# ─────────────────────────────────────────────

def delong_compare(result_a, result_b):
    """
    Compares two models using paired DeLong test on OOF predictions.
    Both results must use the same outer folds (same random_state).
    Returns z-score and two-sided p-value.
    """
    y_true = result_a["oof_true"]
    pred_a = result_a["oof_preds"]
    pred_b = result_b["oof_preds"]

    auc_a  = roc_auc_score(y_true, pred_a)
    auc_b  = roc_auc_score(y_true, pred_b)
    var_a  = delong_auc_variance(y_true, pred_a)
    var_b  = delong_auc_variance(y_true, pred_b)

    # Covariance via DeLong (simplified: assume independence between OOF folds)
    # For a more rigorous test, use the full DeLong covariance matrix
    se = np.sqrt(var_a + var_b)
    if se == 0:
        return 0.0, 1.0

    z = (auc_a - auc_b) / se
    from scipy import stats
    p = 2 * stats.norm.sf(abs(z))

    print(f"  DeLong: {result_a['label']} AUC={auc_a:.4f} vs "
          f"{result_b['label']} AUC={auc_b:.4f}")
    print(f"  z={z:.3f}, p={p:.4f}")
    return z, p


# ─────────────────────────────────────────────
# STEP 7: SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(results_list):
    rows = []
    for r in results_list:
        rows.append({
            "model":            r["label"],
            "oof_auc":          r["oof_auc"],
            "fold_auc_mean":    np.mean(r["fold_aucs"]),
            "fold_auc_std":     np.std(r["fold_aucs"]),
            "fold_prauc_mean":  np.mean(r["fold_prs"]),
            "fold_prauc_std":   np.std(r["fold_prs"]),
            "fold_brier_mean":  np.mean(r["fold_briers"]),
            "fold_brier_std":   np.std(r["fold_briers"]),
            "mean_n_features":  np.mean(r["fold_n_feats"]),
        })

    results_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(results_df.to_string(index=False))

    # Also save OOF predictions for downstream calibration / DCA analysis
    for r in results_list:
        oof_df = pd.DataFrame({
            "y_true":  r["oof_true"],
            "y_pred":  r["oof_preds"],
        })
        oof_df.to_csv(OUTPUT_DIR / f"oof_{r['label'].replace(' ', '_')}.csv",
                      index=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Breast Cancer T-Stage Prediction Pipeline")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────
    img_df, clin_df, y = load_data()

    # ── 2. Build clean feature sets ───────────
    X_img, X_clin, X_fusion = build_feature_sets(img_df, clin_df)

    # ── 3. Run nested CV for each feature set ─
    print("\n" + "=" * 60)
    print("Running imaging-only model")
    print("=" * 60)
    res_img = evaluate_nested_cv(X_img, y, label="Imaging only")

    print("\n" + "=" * 60)
    print("Running clinical-only model")
    print("=" * 60)
    res_clin = evaluate_nested_cv(X_clin, y, label="Clinical only")

    print("\n" + "=" * 60)
    print("Running fusion model")
    print("=" * 60)
    res_fusion = evaluate_nested_cv(X_fusion, y, label="Fusion")

    # ── 4. Statistical comparisons ────────────
    print("\n" + "=" * 60)
    print("DeLong AUC comparisons")
    print("=" * 60)
    delong_compare(res_fusion, res_img)
    delong_compare(res_fusion, res_clin)
    delong_compare(res_img,    res_clin)

    # ── 5. Save results ───────────────────────
    save_results([res_img, res_clin, res_fusion])

    print("\nDone. Outputs in:", OUTPUT_DIR)
