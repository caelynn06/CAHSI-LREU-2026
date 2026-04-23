"""
Feature Importance Analysis — Surgery Clinical Model (Cohort A)
===============================================================
Runs the same nested CV as surgery_pipeline.py but tracks
permutation importance scores across all outer folds for the
CLINICAL model only (the one going on the poster).

Outputs
-------
  pipeline_outputs/surgery_clinical_importance.csv  — full ranked table
  pipeline_outputs/surgery_stable_features.csv      — selected in >= 4/5 folds
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

OUTER_FOLDS  = 5
INNER_FOLDS  = 5
RANDOM_STATE = 42
STABILITY_MIN_FOLDS = 4
IMPORTANCE_THRESHOLD = 0.001

BASE_CATBOOST_PARAMS = dict(
    loss_function="Logloss", eval_metric="AUC",
    iterations=300, learning_rate=0.05, depth=6,
    verbose=False, random_seed=RANDOM_STATE,
)

# Same 15 clinical features as surgery pipeline
CLINICAL_COMPLETE_FEATURES = [
    "Tumor Characteristics | Staging(Tumor Size)# [T]",
    "Tumor Characteristics | Staging(Nodes)#(Nx replaced by -1)[N]",
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
    "MRI Findings | Lymphadenopathy or Suspicious Nodes",
]

CLEAN_NAMES = {
    "Tumor Characteristics | Staging(Tumor Size)# [T]":        "T stage",
    "Tumor Characteristics | Staging(Nodes)#(Nx replaced by -1)[N]": "N stage",
    "Tumor Characteristics | ER":                              "ER status",
    "Tumor Characteristics | PR":                              "PR status",
    "Tumor Characteristics | HER2":                            "HER2 status",
    "Tumor Characteristics | Mol Subtype":                     "Molecular subtype",
    "Tumor Characteristics | Tumor Grade":                     "Tumour grade",
    "Demographics | Menopause (at diagnosis)":                 "Menopausal status",
    "Demographics | Race and Ethnicity":                       "Race / ethnicity",
    "Demographics | Date of Birth (Days)":                     "Age at diagnosis",
    "MRI Findings | Skin/Nipple Invovlement":                  "Skin/nipple involvement",
    "MRI Findings | Multicentric/Multifocal":                  "Multicentric / multifocal",
    "MRI Findings | Pec/Chest Involvement":                    "Pec/chest involvement",
    "MRI Findings | Contralateral Breast Involvement":         "Contralateral involvement",
    "MRI Findings | Lymphadenopathy or Suspicious Nodes":      "Lymphadenopathy",
}

CATEGORY_MAP = {
    "T stage":                   "Staging",
    "N stage":                   "Staging",
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
    "Lymphadenopathy":           "MRI findings",
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
    print("Surgery Clinical Model — Feature Importance")
    print("=" * 60)

    clin = pd.read_csv(CLINICAL_PATH).set_index("Patient ID")

    neoadj_col = "Neoadjuvant therapy | Received Neoadjuvant Therapy or Not"
    surg_col   = "SURGERY | Definitive Surgery Type"

    mask = (clin[neoadj_col] == 2) & clin[surg_col].notna()
    clin = clin[mask]
    y    = clin[surg_col].astype(int)

    avail = [f for f in CLINICAL_COMPLETE_FEATURES if f in clin.columns]
    X     = clin[avail].copy()
    X.columns = [CLEAN_NAMES.get(c, c) for c in X.columns]

    print(f"Patients : {len(y)}")
    print(f"Features : {X.shape[1]}")
    print(f"Mastectomy: {y.sum()}  BCS: {(y==0).sum()}")

    feat_names = X.columns.tolist()
    imp_matrix  = pd.DataFrame(0.0,
                               index=feat_names,
                               columns=range(1, OUTER_FOLDS+1))
    sel_matrix  = pd.DataFrame(0,
                               index=feat_names,
                               columns=range(1, OUTER_FOLDS+1))

    cat_cols = get_cat_cols(X)
    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS,
                                shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, te_idx) in enumerate(
            outer_skf.split(X, y), start=1):

        print(f"\nOuter fold {fold}/{OUTER_FOLDS} ...")
        X_train = X.iloc[tr_idx]
        X_test  = X.iloc[te_idx]
        y_train = y.iloc[tr_idx]
        y_test  = y.iloc[te_idx]

        # Inner loop — permutation importance on training data
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

        # Record importance and selection
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
        cat_idx = [X_train[selected].columns.get_loc(c)
                   for c in cat_cols if c in selected]
        X_tr = prep(X_train[selected], [c for c in cat_cols if c in selected])
        X_te = prep(X_test[selected],  [c for c in cat_cols if c in selected])
        model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
        model.fit(X_tr, y_train, cat_features=cat_idx)
        proba = model.predict_proba(X_te)[:,1]
        auc   = roc_auc_score(y_test, proba)
        print(f"  AUC={auc:.4f}  selected={len(selected)}")

    # Aggregate
    results = pd.DataFrame({
        "feature":         feat_names,
        "clean_name":      feat_names,
        "category":        [CATEGORY_MAP.get(f, "Other") for f in feat_names],
        "mean_importance": imp_matrix.mean(axis=1).values,
        "std_importance":  imp_matrix.std(axis=1).values,
        "fold_count":      sel_matrix.sum(axis=1).values,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    results["rank"] = results.index + 1

    # Save
    results.to_csv(OUTPUT_DIR/"surgery_clinical_importance.csv", index=False)
    stable = results[results["fold_count"] >= STABILITY_MIN_FOLDS]
    stable.to_csv(OUTPUT_DIR/"surgery_stable_features.csv", index=False)

    # Print
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE RESULTS")
    print("="*60)
    print(f"\nAll features ranked:")
    print(f"{'Rank':<5} {'Feature':<30} {'Mean imp':>9} {'Std':>7} {'Folds':>6} {'Category'}")
    print("-"*75)
    for _, row in results.iterrows():
        print(f"{int(row['rank']):<5} {row['feature']:<30} "
              f"{row['mean_importance']:>9.5f} "
              f"{row['std_importance']:>7.5f} "
              f"{int(row['fold_count']):>6}   {row['category']}")

    print(f"\nStable features (>= {STABILITY_MIN_FOLDS}/5 folds): {len(stable)}")
    for _, row in stable.iterrows():
        print(f"  {row['feature']}  (fold_count={int(row['fold_count'])},"
              f" mean={row['mean_importance']:.5f})")

    print(f"\nSaved: {OUTPUT_DIR/'surgery_clinical_importance.csv'}")
    print(f"Saved: {OUTPUT_DIR/'surgery_stable_features.csv'}")
