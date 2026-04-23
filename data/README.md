## This project uses the Duke Breast Cancer MRI dataset available through the Cancer Imaging Archive (TCIA). The datasets used in these models include:   

### Clinical and Other Features (XLSX file), which consists of:   
 -  922 patients with newly diagnosed, biopsy-confirmed breast cancer
 -  18 nested features with sub features inside with demographic, pathologic and clinical information

### Imaging Features
- 922 patients MRI imaging features information
- 539 features with radiomic, measurments, and classification
  
### Missing Values and Data Leakage
A leakage audit was conducted prior to modelling, excluding treatment variables, surgical outcomes, and post-diagnosis follow-up data to ensure predictors reflected only information available at diagnosis.
Synthetic data augmentation was deemed impractical given high missingness relative to sample size, as generative approaches require dense feature spaces and risk amplifying noise.

To reproduce this work, download the dataset from TCIA and place the files in the data/ folder.

Saha, A., Harowicz, M. R., Grimm, L. J., Weng, J., Cain, E. H., Kim, C. E., Ghate, S. V., Walsh, R., & Mazurowski, M. A. (2021). Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.e3sv-re93
