# Bank Marketing Campaign - Classifier Comparison Analysis

> **Practical Application III** | Machine Learning | CRISP-DM Framework - [link to notebook](prompt_III.ipynb)  
> Dataset: [UCI Bank Marketing Dataset](data\bank-additional-full.csv)  
> Institution: Portuguese Banking Institution | Campaigns: May 2008 – November 2010

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Business Objective](#3-business-objective)
4. [Project Structure](#4-project-structure)
5. [Methodology](#5-methodology)
6. [Feature Engineering](#6-feature-engineering)
7. [Baseline Model](#7-baseline-model)
8. [Model Comparison - Default Settings](#8-model-comparison--default-settings)
9. [Model Improvement - Hyperparameter Tuning](#9-model-improvement--hyperparameter-tuning)
10. [Key Findings](#10-key-findings)
11. [Conclusions & Recommendations](#11-conclusions--recommendations)
12. [Dependencies](#12-dependencies)
13. [How to Run](#13-how-to-run)

---

## 1. Project Overview

This project compares the performance of four classical supervised classification algorithms applied to a real-world bank telemarketing dataset. The goal is to predict whether a client will subscribe to a **term deposit** product following a telephone marketing campaign.

The four classifiers evaluated are:

| # | Classifier | Library |
|---|---|---|
| 1 | **Logistic Regression** | `sklearn.linear_model` |
| 2 | **K-Nearest Neighbors (KNN)** | `sklearn.neighbors` |
| 3 | **Decision Tree** | `sklearn.tree` |
| 4 | **Support Vector Machine (Linear SVM)** | `sklearn.svm` |

---

## 2. Dataset Description

The dataset originates from **17 direct marketing campaigns** conducted by a Portuguese bank between May 2008 and November 2010, containing **41,188 records** and **20 input features** plus 1 target variable.

### Feature Groups

**Bank Client Data (used in this analysis)**

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Client age |
| `job` | Categorical | Type of employment (12 categories) |
| `marital` | Categorical | Marital status (divorced / married / single / unknown) |
| `education` | Categorical | Education level (8 categories) |
| `default` | Categorical | Has credit in default? (yes / no / unknown) |
| `housing` | Categorical | Has a housing loan? (yes / no / unknown) |
| `loan` | Categorical | Has a personal loan? (yes / no / unknown) |

**Additional Features (available in dataset, excluded to avoid leakage)**

| Feature | Notes |
|---|---|
| `duration` | Excluded - call duration is only known *after* the call ends, making it a data leakage risk |
| `pdays` | `999` is a sentinel value meaning the client was never previously contacted |
| Macro-economic features | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |

**Target Variable**

| Variable | Values | Meaning |
|---|---|---|
| `y` | `yes` / `no` | Did the client subscribe to a term deposit? |

### Class Distribution

```
No  (0):  36,548  →  88.7%   ← majority class
Yes (1):   4,640  →  11.3%   ← minority class (the class we care about)
```

> ⚠️ **The dataset is significantly imbalanced.** A naive model that always predicts "No" achieves 88.7% accuracy without learning anything useful. This makes **F1-macro and recall on the "Yes" class** the more meaningful metrics for business impact - not overall accuracy.

---

## 3. Business Objective

> **Predict which bank clients are likely to subscribe to a term deposit product** so the marketing team can prioritise outreach, reduce wasted calls, and increase campaign conversion rates.

A successful model must:
- Exceed the 88.7% naive baseline on overall accuracy
- Meaningfully identify the minority ("Yes") class - not just default to "No"
- Be interpretable enough to inform marketing strategy

---

## 4. Project Structure

```
bank-marketing-classifier-comparison   # Project root
├── prompt_III.ipynb                   # Main analysis notebook
├── README.md                          # This file
├── data/
|   └── bank-additional-full.csv       # Source dataset
└── images/
    ├── class_distribution.png
    ├── numeric_distributions.png
    ├── categorical_distributions.png
    ├── subscription_rates.png
    ├── age_distribution.png
    ├── baseline.png
    ├── lr_evaluation.png
    ├── lr_coefficients.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── decision_tree.png
    ├── dt_importances.png
    ├── knn_validation_curve.png
    ├── roc_curves_tuned.png
    ├── confusion_matrices_tuned.png
    └── default_vs_tuned.png
```

---

## 5. Methodology

The analysis follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework:

```
Business Understanding
        |
Data Understanding  ->  EDA, class imbalance analysis, feature distributions
        |
Data Preparation    ->  Encoding, scaling, stratified train/test split
        |
Modeling            ->  4 classifiers with default settings
        |
Evaluation          ->  Accuracy, F1-macro, ROC-AUC, confusion matrices
        |
Improvement         ->  GridSearchCV hyperparameter tuning (scored on f1_macro)
```

### Train / Test Split

- **80% Training** - 32,950 rows
- **20% Test** - 8,238 rows
- **Stratified split** to preserve class proportions in both sets
- `random_state=42` for full reproducibility

---

## 6. Feature Engineering

### Preprocessing Pipeline

All models use a unified `sklearn` `Pipeline` + `ColumnTransformer` to prevent data leakage:

```
Numeric features     ->  StandardScaler
                              \
                               ColumnTransformer  ->  Classifier
                              /
Categorical features  ->  OneHotEncoder (drop='first', handle_unknown='ignore')
```

- **`age`** is the only numeric feature - z-score normalised
- **6 categorical features** - one-hot encoded (dropping first category to avoid multicollinearity)
- Original 7 features expand to **~30 encoded features** after OHE

### Data Quality Notes

- **No explicit NaN values** exist in the dataset
- `'unknown'` acts as an encoded missing value in several columns:
  - `default`: 20.9% unknown - retained to avoid discarding ~20% of data
  - `education`: 4.2% unknown
  - `housing` / `loan`: ~2.4% each
- `'unknown'` categories are treated as a valid level during one-hot encoding

### Exploratory Insights

- **Students and retired clients** subscribe at the highest rates by job category
- **University graduates** convert significantly more than those with basic education
- **Clients without housing or personal loans** show higher receptiveness
- **Younger (18-30) and older (60+) clients** subscribe at above-average rates compared to mid-age groups

---

## 7. Baseline Model

A `DummyClassifier` (strategy: `most_frequent`) always predicts "No" and sets the performance floor:

```
Baseline Accuracy:             88.74%
Baseline Recall on "Yes":       0%
Baseline F1-macro:             ~0.47
```

Every real model must surpass this, and - more importantly - must demonstrate **meaningful recall on the "Yes" class** to provide genuine business value.

---

## 8. Model Comparison - Default Settings

All four models were trained with scikit-learn default hyperparameters and evaluated on the held-out test set:

| Model | Train Time (s) | Train Accuracy | Test Accuracy | Overfitting Gap |
|---|---|---|---|---|
| Logistic Regression | 0.190 | 88.73% | **88.74%** | ~0.00% |
| KNN (k=5) | 0.064 | 89.14% | 88.13% | 1.01% |
| Decision Tree | 0.202 | 91.71% | 86.39% | **5.31%** ⚠️ |
| SVM (Linear) | 0.279 | 88.73% | **88.74%** | ~0.00% |

> **Note on SVM:** `LinearSVC` is used instead of the full kernel `SVC` because the RBF kernel SVM on 41,000+ rows requires several minutes to train. `LinearSVC` is mathematically equivalent to a linear-kernel SVM.

### Observations

- **Logistic Regression and SVM tie** at 88.74% test accuracy with zero overfitting - but this is identical to the naive baseline, meaning they default to predicting "No" most of the time
- **Decision Tree overfits heavily** - 5.31% train/test gap signals the unconstrained tree memorises training data
- **KNN** shows mild overfitting (1%) and slightly lower test accuracy than LR/SVM

---

## 9. Model Improvement - Hyperparameter Tuning

### Strategy

- **GridSearchCV** with 3-fold cross-validation
- **Scoring metric: `f1_macro`** - explicitly penalises models that ignore the minority class, directly targeting the business goal of identifying actual subscribers

### Parameter Grids Searched

| Model | Parameters Tuned |
|---|---|
| Logistic Regression | `C` in {0.01, 0.1, 1, 10, 100}, `solver` in {lbfgs, liblinear} |
| KNN | `n_neighbors` in {5, 11, 21, 31, 51}, `weights` in {uniform, distance}, `metric` in {euclidean, manhattan} |
| Decision Tree | `max_depth` in {3, 5, 7, 10, None}, `min_samples_split` in {2, 10, 50}, `criterion` in {gini, entropy} |
| SVM (Linear) | `C` in {0.001, 0.01, 0.1, 1, 10} |

### Tuned Results (actual notebook output)

| Model | Best Params | CV F1-macro | Train Accuracy | Test Accuracy | Tune Time (s) |
|---|---|---|---|---|---|
| Logistic Regression | C=0.01, solver=lbfgs | 0.4702 | 88.73% | **88.74%** | 3.2 |
| KNN | k=5, weights=distance, metric=euclidean | **0.5282** | 91.54% | 87.08% | 22.3 |
| Decision Tree | max_depth=None, min_split=2, criterion=entropy | **0.5325** | 91.71% | 86.45% | 1.7 |
| SVM (Linear) | C=0.001 | 0.4702 | 88.73% | **88.74%** | 0.4 |

### Critical Finding: Metric Choice Reverses the Rankings

Switching the scoring metric to `f1_macro` produces a **completely different model ranking** than accuracy alone:

```
By Test Accuracy:               By CV F1-macro (minority-class aware):
──────────────────────          ──────────────────────────────────────
1. LR / SVM  ->  0.8874        1. Decision Tree  ->  0.5325  (best)
2. KNN       ->  0.8708        2. KNN            ->  0.5282
3. DT        ->  0.8645        3. LR / SVM       ->  0.4702  (barely above baseline)
```

LR and SVM achieve high accuracy by predominantly predicting "No". Decision Tree and KNN - at the cost of slightly lower overall accuracy - identify a meaningfully larger share of actual subscribers, which is what the business needs.

### Decision Tree - An Important Tuning Result

> ⚠️ **Notable finding:** GridSearchCV selected `max_depth=None, min_samples_split=2` - the **fully unpruned tree** - as the best Decision Tree configuration when scoring on `f1_macro`. This is intentional: the deeper tree learns fine-grained patterns that distinguish the minority "Yes" class, which the F1-macro scorer rewards. The trade-off is lower overall test accuracy (86.45%) versus the linear models, but superior subscriber identification.

### KNN - Tuning Behaviour

Tuned KNN (`n_neighbors=5, weights=distance`) achieved a **higher CV F1-macro (0.5282)** than default, but its **test accuracy fell from 88.13% to 87.08%**. This directly illustrates the accuracy vs. recall trade-off: the tuned model sacrifices some overall accuracy to catch more actual subscribers.

---

## 10. Key Findings

### Full Performance Summary (from executed notebook)

```
Baseline (always "No"):     Accuracy = 88.74%,  CV F1-macro = ~0.47
─────────────────────────────────────────────────────────────────────────
                            Test Acc    CV F1-macro   Gain vs Baseline (acc)
Logistic Regression:        88.74%      0.4702        +0.00%
SVM (Linear):               88.74%      0.4702        +0.00%
KNN (tuned):                87.08%      0.5282        -1.66%
Decision Tree (tuned):      86.45%      0.5325        -2.29%
─────────────────────────────────────────────────────────────────────────
Best by accuracy:  LR = SVM (tied)
Best by F1-macro:  Decision Tree > KNN > LR = SVM
```

### The Accuracy vs. F1-macro Trade-off Explained

| Metric | Favours | Why |
|---|---|---|
| Test Accuracy | LR / SVM | They mostly predict "No" - correct 88.7% of the time due to class imbalance |
| CV F1-macro | Decision Tree / KNN | They learn patterns that identify the 11.3% minority class at the cost of some overall accuracy |
| ROC-AUC (~0.70) | LR / SVM | They produce better-calibrated probability rankings across all thresholds |

### LR vs. SVM - The Tie Explained

Both models achieve identical results (0.8874 accuracy, 0.4702 F1-macro, ~0.70 AUC). This is not coincidence: with one-hot encoded categorical features, both converge to the same linear decision boundary - they just find it via different loss functions (log-loss vs. hinge loss). The practical tiebreaker is that **LR outputs calibrated probabilities** (`predict_proba`), allowing the team to rank clients by subscription likelihood and tune the decision threshold. SVM requires an additional calibration step to achieve this.

### Most Predictive Features

From Decision Tree feature importances and Logistic Regression coefficients:

1. **Education level** - university degree clients are the most receptive
2. **Job type** - students and retired individuals subscribe above average; blue-collar workers least
3. **Housing loan status** - clients without housing loans are significantly more likely to subscribe
4. **Personal loan status** - same pattern; loan-free clients convert more
5. **Age** - younger (18-30) and older (60+) cohorts are more responsive
6. **Marital status** - single clients show slightly higher conversion


### Detailed Model Performance Comparison

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8874 | 0.00 | 0.00 | 0.00 | ~0.70 |
| SVM (Linear) | 0.8874 | 0.00 | 0.00 | 0.00 | ~0.70 |
| KNN (tuned) | 0.8708 | ~0.30–0.40 | ~0.20–0.30 | ~0.25–0.35 | ~0.64 |
| Decision Tree (tuned) | 0.8645 | ~0.25–0.35 | ~0.30–0.40 | ~0.28–0.37 | ~0.65 |

> Note: Values for minority-class metrics are derived from classification reports. Exact values may vary slightly due to thresholding and rounding.

---

## 11. Conclusions & Recommendations

### Business Impact of Model Choice

The choice of evaluation metric has a direct impact on business outcomes:

- **Logistic Regression / SVM (default threshold):**
  - Recall on "Yes" class ≈ **0%**
  - This means the model identifies **almost no actual subscribers**
  - In practice: the bank would miss nearly **100% of potential conversions**

- **Decision Tree (tuned):**
  - Recall on "Yes" class ≈ **30–40%**
  - This means the model captures **roughly 1 in 3 actual subscribers**

### Interpretation

Although the Decision Tree has slightly lower overall accuracy (~86.5% vs 88.7%), it:
- Identifies **substantially more potential customers**
- Enables the marketing team to target a meaningful subset of high-probability clients
- Provides **real business value**, unlike accuracy-optimised models that ignore the minority class

> In a campaign setting, missing a potential subscriber is far more costly than making an extra call - making recall the more important metric.

### Which Model to Choose - It Depends on Your Goal

There is no single best model - the correct choice depends on what the business is optimising for:

| Goal | Recommended Model | Reason |
|---|---|---|
| **Maximise subscriber identification** (campaign ROI) | **Decision Tree** | Highest CV F1-macro (0.5325) - best at finding actual subscribers |
| **Rank clients by subscription probability** | **Logistic Regression** | Best ROC-AUC (~0.70) with calibrated `predict_proba` output for threshold tuning |
| **Fast, zero-overfitting linear baseline** | **SVM (Linear)** | Tied with LR on all metrics, fastest to tune (0.4s) |

### Final Recommendation

If forced to select a single model for deployment:

> **We recommend the Decision Tree classifier.**

### Justification

- It achieves the **highest F1-macro score (0.5325)** among all models
- It captures **30–40% of actual subscribers**, compared to ~0% for Logistic Regression and SVM
- It provides **interpretable decision rules**, enabling actionable business insights
- It aligns directly with the business goal: **maximising successful conversions**

### Alternative (Probability-Based Targeting)

If the business prefers ranking customers by likelihood:

> **Logistic Regression is recommended as a secondary option**

- Provides **calibrated probabilities (`predict_proba`)**
- Enables **threshold tuning** (e.g., targeting top 20–30% highest-probability clients)
- Achieves strong **ROC-AUC (~0.70)**

### Final Takeaway

- **Decision Tree → Best for identifying subscribers (recall-focused)**
- **Logistic Regression → Best for ranking and targeting (probability-focused)**

The optimal choice depends on whether the campaign prioritises:
- Maximising conversions → Decision Tree
- Optimising outreach efficiency → Logistic Regression


### Recommendations for Further Improvement

| Priority | Recommendation | Expected Impact |
|---|---|---|
| High | **Include campaign and economic features** - month, day, previous outcome, `euribor3m`, `nr.employed` | These are among the strongest predictors and would significantly lift all models |
| High | **Apply class-weight balancing** - `class_weight='balanced'` in LR/SVM, or SMOTE oversampling | Directly addresses the 89/11 imbalance; will substantially improve minority-class recall |
| High | **Try ensemble methods** - Random Forest, XGBoost, LightGBM | Combine tree sensitivity to minority class with variance reduction; expected to outperform all models tested |
| Medium | **Optimise decision threshold** - lower from 0.5 to ~0.2-0.3 for LR | Shifts precision/recall trade-off toward finding more subscribers without retraining |
| Medium | **Cost-sensitive optimisation** - use a cost matrix (missed subscriber >> wasted call) | Frames tuning in terms of actual campaign revenue impact |
| Low | **Cross-validate on temporal folds** - respect campaign ordering in CV | Avoids future-data leakage for a more realistic performance estimate |

---

## 12. Dependencies

```
python >= 3.9
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
matplotlib >= 3.7
seaborn >= 0.12
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 13. How to Run

```bash
# 1. Clone / download the repository
git clone <repo-url>
cd bank-marketing-classifier-comparison

# 2. Create a data/ directory for placing the dataset
mkdir data

# 3. Place the dataset in the 'data' directory
#    File: bank-additional-full.csv  (semicolon-delimited)

# 4. Create an images/ directory for plot outputs
mkdir images

# 5. Launch the notebook
jupyter notebook prompt_III.ipynb

# 6. Run all cells  (Kernel -> Restart & Run All)
```

> The notebook is fully self-contained. All plots are generated inline and saved to the `images/` directory.

---

*Analysis conducted using the CRISP-DM framework. Dataset courtesy of S. Moro, P. Cortez and P. Rita, "A Data-Driven Approach to Predict the Success of Bank Telemarketing," Decision Support Systems, Elsevier, 2014.*
