"""
Feature Importance & Biomarker Discovery Analysis
==================================================
Builds on the nested CV pipeline. For each outer fold,
records which features were selected and their permutation
importance scores. Aggregates across all folds to find:

  1. Stable features  — selected in 4 or 5 of 5 folds
  2. Importance ranking — mean ± std across folds
  3. Feature category — kinetics / texture / morphology /
                        spatial-stats / spectral

Outputs
-------
  feature_importance_results.csv  — full ranked table
  stable_biomarkers.csv           — fold_count >= 4 only
  figure_feature_importance.png   — publication bar chart
  figure_stability_heatmap.png    — fold-by-fold heatmap
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG  (must match your main pipeline)
# ─────────────────────────────────────────────

IMAGING_PATH  = "imaging_only_with_target.csv"
CLINICAL_PATH = "clean_clinical_rebuilt.csv"
OUTPUT_DIR    = Path("pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTER_FOLDS    = 5
INNER_FOLDS    = 5
RANDOM_STATE   = 42
IMPORTANCE_THRESHOLD = 0.001

# Minimum number of outer folds a feature must appear in
# to be called "stable"
STABILITY_MIN_FOLDS = 4

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
# FEATURE CATEGORY MAP
# ─────────────────────────────────────────────

CATEGORY_RULES = [
    # (substring_list, category_label, hex_color)
    (["SER_map", "Peak_SER", "SER_Total", "SER_Partial", "SER_Washout",
      "Peak_PE", "PE_map"],
     "SER / PE kinetics", "#1D9E75"),

    (["WashinRate", "Washout_rate", "washin", "washout",
      "Uptake_rate", "Uptake", "Enhancement",
      "Max_Enhancement", "Time_to_Peak",
      "Grouping_based_mean_of_peak",
      "Grouping_based_mean_of_washin",
      "Grouping_based_mean_of_washout",
      "Grouping_based_variance_of"],
     "Enhancement kinetics", "#0F6E56"),

    (["Autocorrelation", "Contrast_", "Correlation",
      "Cluster_Prominence", "Cluster_Shade",
      "Dissimilarity", "Energy_", "Entropy_",
      "Homogeneity", "Max_probability", "Max_Probability",
      "Sum_of_squares", "sum_average", "sum_variance",
      "sum_entropy", "difference_entropy",
      "information_measure", "inverse_difference",
      "Inv_Diff", "Inv_diff"],
     "Texture (GLCM)", "#534AB7"),

    (["DFT_CoeffMap", "DHOG", "DLBP", "F1_DT"],
     "Spectral / HOG / LBP", "#D85A30"),

    (["globalMorans", "EnhancementCluster",
      "Variance_of_RGH", "Margin_Gradient",
      "Variance_of_Uptake", "Change_in_variance"],
     "Spatial statistics", "#BA7517"),

    (["TumorMajorAxisLength", "Volume_cu_mm",
      "Median_solidity", "Median_Elongation",
      "Median_Euler", "BEVR", "BEDR", "MF_", "ASD_"],
     "Morphology", "#888780"),

    (["BreastVol", "tissueVol", "breastDensity",
      "Ratio_Tissue_vol"],
     "Background parenchymal", "#D4537E"),
]

def categorize_feature(name):
    for substrings, label, color in CATEGORY_RULES:
        if any(s in name for s in substrings):
            return label, color
    return "Other", "#B4B2A9"


# ─────────────────────────────────────────────
# LEAKAGE-FREE IMAGING FEATURES
# ─────────────────────────────────────────────

IMAGING_DROP = [
    "TumorMajorAxisLength_mm",
    "Volume_cu_mm_Tumor",
    "early_stage",
]

CLINICAL_DROP_EXACT = [
    "Tumor Characteristics | Staging(Tumor Size)# [T]",
    "Tumor Characteristics | Staging(Nodes)#(Nx replaced by -1)[N]",
    "Tumor Characteristics | Staging(Metastasis)#(Mx -replaced by -1)[M]",
    "Mammography Characteristics | Tumor Size (cm)",
    "US features | Tumor Size (cm)",
    "MRI Findings | Lymphadenopathy or Suspicious Nodes",
    "early_stage",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_cat_cols(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def prep_for_catboost(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].fillna("Missing").astype(str)
    for c in [x for x in df.columns if x not in cat_cols]:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    img  = pd.read_csv(IMAGING_PATH).set_index("Patient ID")
    clin = pd.read_csv(CLINICAL_PATH).set_index("Patient ID")

    shared = img.index.intersection(clin.index)
    img    = img.loc[shared]
    clin   = clin.loc[shared]

    valid  = clin["early_stage"].notna()
    clin   = clin[valid]
    img    = img[img.index.isin(clin.index)]

    y = clin["early_stage"].astype(int)

    drop_img  = [c for c in img.columns  if c in IMAGING_DROP]
    drop_clin = [c for c in clin.columns if c in CLINICAL_DROP_EXACT]
    drop_clin += [c for c in clin.columns
                  if any(c.endswith(s) for s in ["__is_NC","__is_NP","__is_NA"])]
    drop_clin += [c for c in clin.columns if c.startswith("has_")]

    X = img.drop(columns=drop_img, errors="ignore")

    print(f"Patients : {len(y)}")
    print(f"Features : {X.shape[1]}")
    print(f"Class    : {y.value_counts().to_dict()}")
    return X, y

# ─────────────────────────────────────────────
# INNER-LOOP FEATURE SELECTION
# ─────────────────────────────────────────────

def select_features_inner(X_train, y_train):
    cat_cols = get_cat_cols(X_train)
    X_prep   = prep_for_catboost(X_train, cat_cols)
    cat_idx  = [X_prep.columns.get_loc(c) for c in cat_cols]
    feat_names = X_prep.columns.tolist()
    all_imp    = []

    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    for tr, va in skf.split(X_prep, y_train):
        model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
        model.fit(X_prep.iloc[tr], y_train.iloc[tr], cat_features=cat_idx)
        res = permutation_importance(
            model, X_prep.iloc[va], y_train.iloc[va],
            n_repeats=3, random_state=RANDOM_STATE, scoring="roc_auc"
        )
        all_imp.append(res.importances_mean)

    mean_imp  = np.array(all_imp).mean(axis=0)
    selected  = [f for f, v in zip(feat_names, mean_imp)
                 if v > IMPORTANCE_THRESHOLD]

    if len(selected) < 20:
        top20 = np.argsort(mean_imp)[::-1][:20]
        selected = [feat_names[i] for i in top20]

    return selected, dict(zip(feat_names, mean_imp))

# ─────────────────────────────────────────────
# MAIN ANALYSIS — nested CV with importance tracking
# ─────────────────────────────────────────────

def run_importance_analysis(X, y):
    """
    Outer 5-fold CV.
    For each fold:
      - inner loop selects features + records their importance
      - outer fold evaluates model performance
      - records per-fold importance scores for ALL features
        (not just selected ones, so unselected = 0)

    Returns
    -------
    fold_results : list of dicts, one per fold
    importance_matrix : DataFrame (features × folds)
    selection_matrix  : DataFrame (features × folds) — binary
    """
    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)

    all_features     = X.columns.tolist()
    # importance_matrix[feature][fold] = mean permutation importance in that fold
    importance_matrix = pd.DataFrame(0.0,
                                     index=all_features,
                                     columns=range(1, OUTER_FOLDS + 1))
    selection_matrix  = pd.DataFrame(0,
                                     index=all_features,
                                     columns=range(1, OUTER_FOLDS + 1))
    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(
            outer_skf.split(X, y), start=1):

        print(f"\nOuter fold {fold}/{OUTER_FOLDS} ...")

        X_train = X.iloc[tr_idx]
        X_test  = X.iloc[te_idx]
        y_train = y.iloc[tr_idx]
        y_test  = y.iloc[te_idx]

        # Inner loop: feature selection on training data only
        selected, imp_scores = select_features_inner(X_train, y_train)
        print(f"  Selected {len(selected)} features")

        # Record importance scores for ALL features in this fold
        for feat, score in imp_scores.items():
            if feat in importance_matrix.index:
                importance_matrix.loc[feat, fold] = max(score, 0)

        # Record selection (binary)
        for feat in selected:
            if feat in selection_matrix.index:
                selection_matrix.loc[feat, fold] = 1

        # Train final model on selected features
        cat_cols = get_cat_cols(X_train[selected])
        X_tr = prep_for_catboost(X_train[selected], cat_cols)
        X_te = prep_for_catboost(X_test[selected],  cat_cols)
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]

        model = CatBoostClassifier(**BASE_CATBOOST_PARAMS)
        model.fit(X_tr, y_train, cat_features=cat_idx)

        proba = model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        print(f"  Fold AUC: {auc:.4f}")

        fold_results.append({
            "fold":      fold,
            "auc":       auc,
            "n_features": len(selected),
            "selected":  selected,
        })

    return fold_results, importance_matrix, selection_matrix

# ─────────────────────────────────────────────
# AGGREGATE RESULTS
# ─────────────────────────────────────────────

def aggregate_results(importance_matrix, selection_matrix):
    """
    Build a summary DataFrame with:
      mean_importance, std_importance, fold_count,
      stability_ratio, category, color
    """
    df = pd.DataFrame({
        "feature":          importance_matrix.index,
        "mean_importance":  importance_matrix.mean(axis=1).values,
        "std_importance":   importance_matrix.std(axis=1).values,
        "fold_count":       selection_matrix.sum(axis=1).values,
    })

    df["stability_ratio"] = df["fold_count"] / OUTER_FOLDS
    df["cv_of_importance"] = (
        df["std_importance"] / (df["mean_importance"].abs() + 1e-8)
    )

    cats   = [categorize_feature(f) for f in df["feature"]]
    df["category"] = [c[0] for c in cats]
    df["color"]    = [c[1] for c in cats]

    df = df.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df

# ─────────────────────────────────────────────
# FIGURE 1 — horizontal bar chart (top N features)
# ─────────────────────────────────────────────

def plot_importance_barchart(results_df, top_n=20, outpath=None):
    plot_df = results_df.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]   # reverse so top feature is at top

    fig, ax = plt.subplots(figsize=(9, top_n * 0.42 + 1.2))

    # Clean short name for display
    def shorten(name):
        name = name.replace("Grouping_based_", "GB_")
        name = name.replace("_Tumor", "")
        name = name.replace("_tumor", "")
        name = name.replace("_tissue_T1", "_T1")
        name = name.replace("_tissue_PostCon", "_PostCon")
        name = name.replace("WashinRate_map_", "Washin_")
        name = name.replace("SER_map_", "SER_")
        name = name.replace("PE_map_", "PE_")
        name = name.replace("Mean_norm_", "")
        name = name.replace("Ratio_Tissue_vol_enhancing_more_than_", "Ratio_enh>")
        name = name.replace("_from_PostCon_to_Breast_Vol", "_PostCon/BreastVol")
        name = name.replace("_from_T1_to_Breast_Vol", "_T1/BreastVol")
        if len(name) > 48:
            name = name[:46] + "…"
        return name

    labels = [shorten(f) for f in plot_df["feature"]]
    vals   = plot_df["mean_importance"].values
    errs   = plot_df["std_importance"].values
    colors = plot_df["color"].values

    # Stability indicator: dim bars with fold_count < STABILITY_MIN_FOLDS
    alphas = [1.0 if fc >= STABILITY_MIN_FOLDS else 0.4
              for fc in plot_df["fold_count"].values]

    y_pos = np.arange(len(plot_df))

    for i, (v, e, c, a, fc) in enumerate(
            zip(vals, errs, colors, alphas, plot_df["fold_count"].values)):
        ax.barh(y_pos[i], v, xerr=e, color=c, alpha=a,
                height=0.65, capsize=3,
                error_kw={"elinewidth": 0.8, "ecolor": "#888780"})
        # Fold count badge
        ax.text(v + e + 0.0005, y_pos[i],
                f"{fc}/5", va="center", ha="left",
                fontsize=8, color="#888780")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean permutation importance (AUC)", fontsize=10)
    ax.set_title(f"Top {top_n} imaging features — mean ± SD across outer folds",
                 fontsize=11, pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.6)

    # Legend for categories present in this plot
    seen_cats = plot_df[["category","color"]].drop_duplicates()
    patches = [mpatches.Patch(color=row["color"], label=row["category"])
               for _, row in seen_cats.iterrows()]
    ax.legend(handles=patches, fontsize=8, loc="lower right",
              framealpha=0.9, edgecolor="#D3D1C7")

    # Dim bar explanation
    ax.text(0.01, -0.06,
            "Dimmed bars = selected in < 4/5 folds (unstable)",
            transform=ax.transAxes, fontsize=8, color="#888780")

    plt.tight_layout()
    path = outpath or OUTPUT_DIR / "figure_feature_importance.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# FIGURE 2 — fold-by-fold selection heatmap
# Shows only features selected in >= 2 folds
# ─────────────────────────────────────────────

def plot_stability_heatmap(results_df, selection_matrix, outpath=None):
    # Keep features selected in >= 2 folds, sort by fold_count then mean_imp
    heatmap_df = results_df[results_df["fold_count"] >= 2].copy()
    heatmap_df = heatmap_df.sort_values(
        ["fold_count", "mean_importance"], ascending=[False, False]
    ).head(40)

    feats     = heatmap_df["feature"].tolist()
    sel_slice = selection_matrix.loc[feats]

    def shorten(name):
        name = name.replace("Grouping_based_", "GB_")
        name = name.replace("_Tumor", "").replace("_tumor", "")
        name = name.replace("_tissue_T1", "_T1")
        name = name.replace("_tissue_PostCon", "_PostCon")
        name = name.replace("WashinRate_map_", "Washin_")
        name = name.replace("SER_map_", "SER_").replace("PE_map_", "PE_")
        name = name.replace("Mean_norm_", "")
        if len(name) > 50: name = name[:48] + "…"
        return name

    short_labels = [shorten(f) for f in feats]
    colors_col   = heatmap_df["color"].tolist()

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2,
        figsize=(13, max(6, len(feats) * 0.32 + 1.5)),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # Heatmap
    data = sel_slice.values.astype(float)
    cmap = matplotlib.colors.ListedColormap(["#F1EFE8", "#1D9E75"])
    ax_heat.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="none")

    ax_heat.set_xticks(range(OUTER_FOLDS))
    ax_heat.set_xticklabels([f"Fold {i+1}" for i in range(OUTER_FOLDS)],
                             fontsize=9)
    ax_heat.set_yticks(range(len(feats)))
    ax_heat.set_yticklabels(short_labels, fontsize=8)

    for i, color in enumerate(colors_col):
        ax_heat.get_yticklabels()[i].set_color(color)

    ax_heat.set_title("Feature selection across outer folds\n(green = selected)",
                      fontsize=10, pad=10)
    ax_heat.tick_params(length=0)

    # Grid lines
    for x in np.arange(-0.5, OUTER_FOLDS, 1):
        ax_heat.axvline(x, color="white", linewidth=1)
    for y in np.arange(-0.5, len(feats), 1):
        ax_heat.axhline(y, color="white", linewidth=0.5)

    # Bar chart: mean importance
    bar_vals   = heatmap_df["mean_importance"].values
    bar_errs   = heatmap_df["std_importance"].values
    bar_colors = heatmap_df["color"].values
    y_pos      = np.arange(len(feats))

    ax_bar.barh(y_pos, bar_vals, xerr=bar_errs,
                color=bar_colors, height=0.65,
                capsize=2,
                error_kw={"elinewidth": 0.6, "ecolor": "#B4B2A9"})
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Mean importance", fontsize=9)
    ax_bar.set_title("Importance", fontsize=10, pad=10)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(length=0)
    ax_bar.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    plt.tight_layout(w_pad=0.5)
    path = outpath or OUTPUT_DIR / "figure_stability_heatmap.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# SAVE TABLES
# ─────────────────────────────────────────────

def save_tables(results_df, fold_results):
    # Full ranked table
    out_cols = ["rank", "feature", "category",
                "mean_importance", "std_importance",
                "fold_count", "stability_ratio", "cv_of_importance"]
    full_path = OUTPUT_DIR / "feature_importance_results.csv"
    results_df[out_cols].to_csv(full_path, index=False)
    print(f"Saved: {full_path}")

    # Stable biomarkers only
    stable = results_df[results_df["fold_count"] >= STABILITY_MIN_FOLDS].copy()
    stable_path = OUTPUT_DIR / "stable_biomarkers.csv"
    stable[out_cols].to_csv(stable_path, index=False)
    print(f"Saved: {stable_path}  ({len(stable)} features)")

    # Per-fold summary
    fold_df = pd.DataFrame([
        {"fold": r["fold"], "auc": r["auc"], "n_features": r["n_features"]}
        for r in fold_results
    ])
    fold_df.to_csv(OUTPUT_DIR / "feature_importance_fold_summary.csv", index=False)


# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(results_df, fold_results):
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 60)

    stable = results_df[results_df["fold_count"] >= STABILITY_MIN_FOLDS]
    print(f"\nStable features (selected in >= {STABILITY_MIN_FOLDS}/5 folds): "
          f"{len(stable)}")
    print(f"Total features with any selection: "
          f"{(results_df['fold_count'] > 0).sum()}")

    print("\nTop 20 features:")
    top20 = results_df.head(20)[
        ["rank","feature","category","mean_importance",
         "std_importance","fold_count"]
    ].copy()
    top20["mean_importance"] = top20["mean_importance"].map("{:.5f}".format)
    top20["std_importance"]  = top20["std_importance"].map("{:.5f}".format)
    print(top20.to_string(index=False))

    print("\nStable features by category:")
    cat_counts = stable["category"].value_counts()
    for cat, n in cat_counts.items():
        print(f"  {cat:<35} {n}")

    print("\nPer-fold performance:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: AUC={r['auc']:.4f}, "
              f"n_features={r['n_features']}")

    mean_auc = np.mean([r["auc"] for r in fold_results])
    std_auc  = np.std([r["auc"] for r in fold_results])
    print(f"\n  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Feature Importance & Biomarker Discovery")
    print("=" * 60)

    X, y = load_data()

    print("\nRunning nested CV with importance tracking ...")
    fold_results, importance_matrix, selection_matrix = \
        run_importance_analysis(X, y)

    print("\nAggregating results ...")
    results_df = aggregate_results(importance_matrix, selection_matrix)

    print("\nGenerating figures ...")
    plot_importance_barchart(results_df, top_n=20)
    plot_stability_heatmap(results_df, selection_matrix)

    save_tables(results_df, fold_results)
    print_summary(results_df, fold_results)

    print("\nDone. Files in:", OUTPUT_DIR)
