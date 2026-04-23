"""
Feature Importance — T-stage Clinical Model
============================================
Same structure as surgery_feature_importance.py but for
the T-stage target (early_stage: T0/T1 vs T2+) using all 916 patients.

Outputs
-------
  pipeline_outputs/tstage_clinical_importance.csv
  pipeline_outputs/tstage_stable_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

CLINICAL_PATH = "clean_clinical_rebuilt.csv"
OUTPUT_DIR    = Path("pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTER_FOLDS          = 5
INNER_FOLDS          = 5
RANDOM_STATE         = 42
STABILITY_MIN_FOLDS  = 4
IMPORTANCE_THRESHOLD = 0.001

BASE_CATBOOST_PARAMS = dict(
    loss_function="Logloss", eval_metric="AUC",
    iterations=300, learning_rate=0.05, depth=6,
    verbose=False, random_seed=RANDOM_STATE,
)

# T-stage clinical features — same 12 as clinical-complete model
# NOTE: T stage and N stage are EXCLUDED here because they are
# inputs to the label definition (early_stage = T0/T1 vs T2+)
CLINICAL_COMPLETE_FEATURES = [
    "Tumor Characteristics | ER",
    "Tumor Characteristics | PR",
    "Tumor Characteristics | HER2",
    "Tumor Characteristics | Mol Subtype",
    "Tumor Characteristics | Tumor Grade",
    "Demographics | Menopause (at diagnosis)",
    "Demographics | Race and Ethnicity",
    "Demographics | Date of Birth (Days)",
    "MRI Findings | Skin/Nipple Invovlement",
    "MRI Findings | Multicentric/Multifocal",
    "MRI Findings | Pec/Chest Involvement",
    "MRI Findings | Contralateral Breast Involvement",
]

CLEAN_NAMES = {
    "Tumor Characteristics | ER":                    "ER status",
    "Tumor Characteristics | PR":                    "PR status",
    "Tumor Characteristics | HER2":                  "HER2 status",
    "Tumor Characteristics | Mol Subtype":           "Molecular subtype",
    "Tumor Characteristics | Tumor Grade":           "Tumour grade",
    "Demographics | Menopause (at diagnosis)":       "Menopausal status",
    "Demographics | Race and Ethnicity":             "Race / ethnicity",
    "Demographics | Date of Birth (Days)":           "Age at diagnosis",
    "MRI Findings | Skin/Nipple Invovlement":        "Skin/nipple involvement",
    "MRI Findings | Multicentric/Multifocal":        "Multicentric / multifocal",
    "MRI Findings | Pec/Chest Involvement":          "Pec/chest involvement",
    "MRI Findings | Contralateral Breast Involvement": "Contralateral involvement",
}

CATEGORY_MAP = {
    "ER status":                 "Tumour characteristics",
    "PR status":                 "Tumour characteristics",
    "HER2 status":               "Tumour characteristics",
    "Molecular subtype":         "Tumour characteristics",
    "Tumour grade":              "Tumour characteristics",
    "Menopausal status":         "Demographics",
    "Race / ethnicity":          "Demographics",
    "Age at diagnosis":          "Demographics",
    "Skin/nipple involvement":   "MRI findings",
    "Multicentric / multifocal": "MRI findings",
    "Pec/chest involvement":     "MRI findings",
    "Contralateral involvement": "MRI findings",
}


def get_cat_cols(df):
    return df.select_dtypes(
        include=["object","category","bool"]).columns.tolist()


def prep(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].fillna("Missing").astype(str)
    for c in [x for x in df.columns if x not in cat_cols]:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("T-stage Clinical Model — Feature Importance")
    print("=" * 60)

    clin  = pd.read_csv(CLINICAL_PATH).set_index("Patient ID")
    valid = clin["early_stage"].notna()
    clin  = clin[valid]
    y     = clin["early_stage"].astype(int)

    avail = [f for f in CLINICAL_COMPLETE_FEATURES if f in clin.columns]
    X     = clin[avail].copy()
    X.columns = [CLEAN_NAMES.get(c, c) for c in X.columns]

    print(f"Patients  : {len(y)}")
    print(f"Features  : {X.shape[1]}")
    print(f"Early (T0/T1): {y.sum()}  Late (T2+): {(y==0).sum()}")

    feat_names = X.columns.tolist()
    imp_matrix = pd.DataFrame(0.0, index=feat_names,
                              columns=range(1, OUTER_FOLDS+1))
    sel_matrix = pd.DataFrame(0,   index=feat_names,
                              columns=range(1, OUTER_FOLDS+1))

    cat_cols  = get_cat_cols(X)
    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS,
                                shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, te_idx) in enumerate(
            outer_skf.split(X, y), start=1):

        print(f"\nOuter fold {fold}/{OUTER_FOLDS} ...")
        X_train = X.iloc[tr_idx]
        X_test  = X.iloc[te_idx]
        y_train = y.iloc[tr_idx]
        y_test  = y.iloc[te_idx]

        # Inner loop
        inner_skf = StratifiedKFold(n_splits=INNER_FOLDS,
                                    shuffle=True, random_state=RANDOM_STATE)
        all_imp = []
        for tr, va in inner_skf.split(X_train, y_train):
            cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
            X_tr = prep(X_train.iloc[tr], cat_cols)
            X_va = prep(X_train.iloc[va], cat_cols)
            model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
            model.fit(X_tr, y_train.iloc[tr], cat_features=cat_idx)
            res = permutation_importance(
                model, X_va, y_train.iloc[va],
                n_repeats=5, random_state=RANDOM_STATE,
                scoring="roc_auc"
            )
            all_imp.append(res.importances_mean)

        mean_imp = np.array(all_imp).mean(axis=0)

        for feat, score in zip(feat_names, mean_imp):
            imp_matrix.loc[feat, fold] = max(score, 0)

        selected = [f for f, v in zip(feat_names, mean_imp)
                    if v > IMPORTANCE_THRESHOLD]
        if len(selected) < 6:
            top_idx  = np.argsort(mean_imp)[::-1][:6]
            selected = [feat_names[i] for i in top_idx]

        for feat in selected:
            sel_matrix.loc[feat, fold] = 1

        # Outer evaluation
        sel_cat = [c for c in cat_cols if c in selected]
        X_tr = prep(X_train[selected], sel_cat)
        X_te = prep(X_test[selected],  sel_cat)
        cat_idx = [X_tr.columns.get_loc(c) for c in sel_cat]
        model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
        model.fit(X_tr, y_train, cat_features=cat_idx)
        proba = model.predict_proba(X_te)[:,1]
        auc   = roc_auc_score(y_test, proba)
        print(f"  AUC={auc:.4f}  selected={len(selected)}")

    # Aggregate
    results = pd.DataFrame({
        "feature":         feat_names,
        "category":        [CATEGORY_MAP.get(f, "Other") for f in feat_names],
        "mean_importance": imp_matrix.mean(axis=1).values,
        "std_importance":  imp_matrix.std(axis=1).values,
        "fold_count":      sel_matrix.sum(axis=1).values,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    results.to_csv(OUTPUT_DIR/"tstage_clinical_importance.csv", index=False)
    stable = results[results["fold_count"] >= STABILITY_MIN_FOLDS]
    stable.to_csv(OUTPUT_DIR/"tstage_stable_features.csv", index=False)

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE RESULTS")
    print("="*60)
    print(f"\n{'Rank':<5} {'Feature':<30} {'Mean imp':>9} "
          f"{'Std':>7} {'Folds':>6}  Category")
    print("-"*75)
    for _, row in results.iterrows():
        print(f"{int(row['rank']):<5} {row['feature']:<30} "
              f"{row['mean_importance']:>9.5f} "
              f"{row['std_importance']:>7.5f} "
              f"{int(row['fold_count']):>6}  {row['category']}")

    print(f"\nStable features (>= {STABILITY_MIN_FOLDS}/5 folds): {len(stable)}")
    for _, row in stable.iterrows():
        print(f"  {row['feature']}  "
              f"(folds={int(row['fold_count'])}, "
              f"mean={row['mean_importance']:.5f})")

    print(f"\nSaved: {OUTPUT_DIR/'tstage_clinical_importance.csv'}")
    print(f"Saved: {OUTPUT_DIR/'tstage_stable_features.csv'}")
