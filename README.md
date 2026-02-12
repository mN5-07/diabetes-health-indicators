# CDC Diabetes Health Indicators Prediction

**End-to-end machine learning project** using the CDC BRFSS 2015 Diabetes Health Indicators dataset (~253,680 samples, 21 features) to predict whether an individual has diabetes or prediabetes (`Diabetes_binary`: 0 = no, 1 = yes).

Focus on **interpretability**, realistic handling of severe class imbalance (~86% negative), and public-health relevance. Built with scikit-learn; models serialized for reuse.

## Project Overview

- **Objective**  
  Binary classification: predict `Diabetes_binary` from self-reported survey features (BMI, general health, blood pressure, cholesterol, physical activity, age, etc.).

- **Dataset**  
  CDC BRFSS 2015 Diabetes Health Indicators  
  • UCI Repository ID: 891  
  • Link: [https://archive.ics.uci.edu/dataset/891](https://archive.ics.uci.edu/dataset/891)  
  • Common Kaggle mirror: [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  
  • 253,680 rows, 21 features + target  
  • Severe imbalance: ~13.9% positive class

![Class Distribution](figures/class_distribution_bar.png)  
*Severe class imbalance: ~86.1% no diabetes/prediabetes vs. 13.9% positive cases*

- **Best Model** (on held-out test set)  
  **Logistic Regression** with `class_weight='balanced'`  
  → Selected for excellent interpretability, simplicity, strong recall in screening context, and very competitive performance  
  → XGBoost (no SMOTE) achieves marginally higher AUC and tuned F1, but is less interpretable

- **Test Set Performance** (Logistic Regression)

| Metric                  | @ threshold 0.5 | @ optimal threshold | Notes                                      |
|-------------------------|-----------------|----------------------|--------------------------------------------|
| AUC-ROC                 | 0.820           | —                    | Solid discriminative power                 |
| Recall                  | 76.1%           | 65.4%                | Good sensitivity → realistic trade-off     |
| Precision               | 31.1%           | 35.4%                | Improved with tuning                       |
| F1-Score                | 0.441           | **0.459**            | +0.018 gain from threshold tuning          |
| Accuracy                | 73.2%           | 78.5%                | Better calibrated                          |

![ROC Curve](figures/roc-curve.png)  
*ROC Curve - Logistic Regression (AUC ≈ 0.820)*

![Precision-Recall Curve](figures/precision-recall-curve.png)  
*Precision-Recall Curve - Logistic Regression (Best F1 = 0.459 @ threshold ≈ 0.60)*

![Confusion Matrix](figures/confusion-matrix.png)  
*Confusion Matrix - Logistic Regression (default threshold 0.5)*

→ Threshold tuning (maximizing F1) meaningfully improves F1 and reduces false positives while preserving good recall for screening purposes.

## Key Insights & Top Risk Factors

Coefficients + permutation importance consistently highlight:

1. **GenHlth** (self-reported general health — especially poor/fair categories)  
   → By far the strongest predictor

2. **HighBP** & **HighChol**  
   → Well-established clinical risk factors

3. **BMI**  
   → Strong continuous signal — higher values markedly increase risk

4. **Age** (older categories)  
   → Clear age-related increase

5. **PhysActivity** (lack of) & **DiffWalk**  
   → Inactivity and mobility limitations contribute meaningfully

![Diabetes Rate by GenHlth](figures/diabetes_rate_by_genhlth.png)  
*Diabetes/prediabetes rate rises sharply with poorer self-reported health*

![BMI Distribution by Diabetes Status](figures/bmi_dist_by_diabetes_status.png)  
*Higher median BMI and greater spread among individuals with diabetes/prediabetes*

![Top Logistic Regression Coefficients](figures/top_coefficients_ordinal.png)  
*Top coefficients — GenHlth, Age, BMI, HighBP, HighChol dominate*

**Public health takeaway**  
Self-perceived health status combined with modifiable and clinical factors (hypertension, cholesterol, obesity, inactivity) drive most of the predictive signal — consistent with known diabetes epidemiology.

## Project Structure

```
diabetes-health-indicators/
├── data/                     # Raw + processed .npy files + feature names
├── models/                   # Serialized preprocessor & best model
│   ├── best_diabetes_model_Logistic_Regression.joblib
│   └── diabetes_preprocessor.joblib
├── notebooks/                # Jupyter analysis & modeling steps
│   ├── 01_diabetes_exploration.ipynb
│   ├── 02_diabetes_preprocessing.ipynb
│   ├── 03_diabetes_modeling_evaluation.ipynb
│   └── 04_diabetes_feature_importance_insights.ipynb
├── figures/                  # Exported plots
│   ├── class-distribution-bar.png
│   ├── roc-curve.png
│   ├── precision-recall-curve-logistic.png
│   ├── confusion-matrix.png
│   ├── top_coefficients_ordinal.png
│   ├── diabetes_rate_by_genhlth.png
│   └── bmi_by_diabetes_status.png
├── README.md
└── requirements.txt
```

## Tech Stack

- Python 3.12
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- joblib
- ucimlrepo

## How to Reproduce

1. Clone the repository
2. `pip install -r requirements.txt`
3. Run notebooks in order: 01 → 02 → 03 → 04
4. (Optional) Export plots to `figures/` folder

## Results Summary & Model Comparison

| Model                        | AUC     | F1 @ 0.5 | Best F1 (tuned) | Best Threshold | Recall @ best | Precision @ best | Notes                                      |
|------------------------------|---------|----------|------------------|----------------|---------------|------------------|--------------------------------------------|
| XGBoost (no SMOTE)           | **0.822** | 0.443  | **0.465**        | ~0.66          | 58.4%         | **38.6%**        | Slightly highest AUC & tuned F1            |
| Logistic Regression          | 0.820   | 0.441    | 0.459            | ~0.60          | **65.4%**     | 35.4%            | Best interpretability + higher recall      |
| XGBoost + SMOTE              | 0.822   | 0.289    | 0.459            | ~0.25          | 62.0%         | 36.5%            | SMOTE not helpful                          |
| Random Forest                | 0.791   | 0.235    | 0.429            | ~0.22          | 61.2%         | 33.1%            | Poor default recall                        |

**Key findings**:
- SMOTE provided **no meaningful benefit** — class weighting is sufficient
- Ordinal encoding (for GenHlth, Age, Education, Income) maintained performance, reduced feature count (21 total), and greatly improved interpretability
- Threshold tuning improves F1 by ~0.018–0.022 and gives a better precision/recall balance for screening

## Future Ideas

- Interactive Streamlit risk calculator (with probability + risk level)
- SHAP explanations for tree-based models
- Probability calibration (e.g. Platt scaling)
- Explore newer BRFSS years for validation

**Important disclaimer**  
Educational demonstration only. Not a medical diagnostic tool — do **not** use for clinical decisions.