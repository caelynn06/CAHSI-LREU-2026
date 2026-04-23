![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NSF](https://img.shields.io/badge/Funded%20by-NSF-orange)
# Comparative Analysis of Radiomic and Clinical Features for Breast Cancer Staging and Surgical Decision‑Making Using Machine Learning   
## CAHSI LREU 2026

### Overview
This project investigates whether quantitative MRI radiomic features or structured clinical features are more predictive of two clinically relevant breast cancer outcomes: tumor T-stage (T0/T1 vs. T2+) and definitive surgery type (mastectomy vs. breast-conserving surgery). Gradient boosting classifiers (CatBoost) were trained on clinical features alone, imaging radiomic features alone, and a fusion of both, then evaluated within a nested cross-validation framework.
The central finding is that each modality predicts what it measures: radiomic features substantially outperform clinical features for T-stage prediction, while clinical features outperform radiomics for surgical planning. Fusion models provided no meaningful improvement in either case.

### Methodology
Models were evaluated using **nested 5-fold stratified cross-validation**:
- **Inner loop** — CatBoost trained on inner training folds, permutation importance computed on inner validation folds, top features selected
- **Outer loop** — Final model trained on selected features, evaluated on held-out test fold
- Out-of-fold predictions pooled across all 5 outer folds for final performance estimates
- No information from the test fold influences feature selection or model training at any stage

Three feature sets were compared per outcome: clinical only, imaging only, and fusion.
All predicted probabilities post-hoc calibrated using Platt scaling.
### Results Summary
| Target | Model | N | AUC | 95% CI | ECE | Brier |
|:---|:---|---:|---:|:---:|---:|---:|
| T-stage (T0/T1 vs T2+) | Clinical only | 916 | 0.608 | [0.572, 0.644] | 0.019 | 0.239 |
| T-stage (T0/T1 vs T2+) | Imaging only  | 916 | 0.787 | [0.760, 0.819] | 0.025 | 0.187 |
| T-stage (T0/T1 vs T2+) | Fusion        | 916 | 0.788 | [0.759, 0.817] | 0.048 | 0.190 |
| Surgical planning       | Clinical only | 559 | 0.714 | [0.667, 0.757] | 0.056 | 0.207 |
| Surgical planning       | Imaging only  | 559 | 0.668 | [0.623, 0.715] | 0.032 | 0.224 |
| Surgical planning       | Fusion        | 559 | 0.731 | [0.683, 0.776] | 0.046 | 0.203 |

### Key Findings
- Imaging radiomic features substantially outperform clinical features for T-stage prediction (AUC 0.787 vs. 0.608, p<0.001)
- Clinical features outperform imaging radiomics for surgical planning (AUC 0.714 vs. 0.668)
Fusion models provide no significant improvement over the best single-modality model for either outcome
- The dominant radiomic predictor for T-stage was washin-rate texture heterogeneity; the dominant clinical predictor for surgery was multicentric or multifocal disease
### How to Reproduce
```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data from TCIA and place in data/

# 3. Run T-stage pipeline
python pipeline/staging_pipeline_clean.py

# 4. Run surgery pipeline
python pipeline/surgery_pipeline.py A

# 5. Run feature importance
python pipeline/tstage_clinical_importance.py
python pipeline/surgery_feature_importance.py

# 6. Run calibration
python calibration/calibration_all_models.py

# 7. Generate figures
python figures/poster_fig1_roc_v4.py
python figures/poster_fig2_final_v3.py
python figures/poster_fig3_calibration_v2.py
python figures/poster_fig4_table_v3.py
python figures/poster_fig5_distributions_v2.py
```
### Acknowledgements
This material is based upon work supported by the National Science Foundation under Grant Numbers CNS-2137781 and HRD-1834620. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

