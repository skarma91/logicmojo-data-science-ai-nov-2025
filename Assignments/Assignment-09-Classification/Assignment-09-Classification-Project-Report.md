# Project Report: Bank Marketing Classification

## 1. Problem Framing & Evaluation Metrics

### Problem Statement
A Portuguese banking institution conducted direct phone marketing campaigns to promote term deposit subscriptions. The goal is to build a predictive model that determines whether a client will subscribe to a term deposit (`yes`/`no`) based on demographic, economic, and campaign-related features.

This is a **binary classification** problem with significant **class imbalance** — approximately 88% of clients did not subscribe vs. 12% who did.

### Choice of Evaluation Metrics
Given the class imbalance, **accuracy alone is misleading** — a naive model predicting "no" for every client would achieve ~88% accuracy while being completely useless. We therefore evaluate models using multiple metrics:

- **ROC-AUC** (primary metric): Measures the model's discriminative ability across all classification thresholds. Chosen as the primary metric because it is threshold-independent and robust to class imbalance.
- **F1-Score**: The harmonic mean of precision and recall, providing a single measure that balances both concerns for the minority class.
- **Precision**: Important because false positives waste campaign resources (calling clients who won't subscribe).
- **Recall**: Critical because false negatives represent missed revenue opportunities (failing to contact potential subscribers).

---

## 2. EDA Insights

### Class Imbalance
The target variable is heavily imbalanced: 36,548 clients (88.7%) did not subscribe vs. 4,640 (11.3%) who did — a ratio of approximately 7.9:1. This imbalance necessitates specialized modeling techniques.

### Key Feature-Target Relationships
- **Age**: Subscribers skew toward younger (students, under 30) and older (retired, over 60) demographics. The 30–55 working-age group is harder to convert.
- **Job type**: Students and retired individuals show the highest subscription rates (~25–30%), while blue-collar and services workers show the lowest (~7%).
- **Contact type**: Cellular contacts have roughly double the subscription rate of telephone contacts.
- **Month**: March, December, September, and October show higher subscription rates, while May (the month with the most contacts) shows a low rate — suggesting campaign fatigue.
- **Previous campaign outcome (`poutcome`)**: Clients with a successful previous outcome have a dramatically higher subscription rate (~65%), making this one of the strongest predictors.

### Macroeconomic Features
The five socio-economic features (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) are strongly inter-correlated and show clear relationships with subscription outcomes. Lower Euribor rates and employment variation rates correlate with higher subscription rates — clients are more likely to lock in term deposits when economic conditions favor savings.

### Note on Duration
The `duration` feature (last contact call duration) is the strongest single predictor but was **excluded** from modeling because it is only known after the call ends and thus cannot be used for pre-campaign prediction.

---

## 3. Preprocessing Summary

### Missing Values
Missing values are encoded as `"unknown"` in categorical columns. The most affected features are:
- `default` (~20% unknown)
- `education` (~4% unknown)
- `job`, `marital`, `housing`, `loan` (~1–2% unknown)

**Strategy**: We retained `"unknown"` as its own category during one-hot encoding rather than imputing, since the missingness itself may carry signal (e.g., clients who refuse to disclose credit default status may behave differently).

### Feature Encoding
- **Categorical features** (10 columns): One-hot encoded using `OneHotEncoder` with `drop='if_binary'` to avoid multicollinearity for binary features (e.g., `contact`, `housing`, `loan`).
- **Numeric features** (9 columns, after dropping `duration`): Standardized using `StandardScaler` to zero mean and unit variance.

### Train/Test Split
- 80/20 stratified split (preserving the class distribution).
- Training set: 32,950 samples; Test set: 8,238 samples.

---

## 4. Model Comparison

### Baseline Models

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | 0.901    | 0.688     | 0.219  | 0.332 | 0.801   |
| Decision Tree       | 0.839    | 0.300     | 0.320  | 0.310 | 0.616   |
| Naive Bayes         | 0.807    | 0.319     | 0.627  | 0.423 | 0.773   |

### Advanced Models (with Imbalance Handling)

| Model                       | Accuracy | Precision | Recall | F1    | ROC-AUC |
|-----------------------------|----------|-----------|--------|-------|---------|
| Random Forest (balanced)    | 0.895    | 0.571     | 0.280  | 0.376 | 0.781   |
| Gradient Boosting           | 0.900    | 0.627     | 0.284  | 0.391 | 0.813   |
| XGBoost (weighted)          | 0.846    | 0.390     | 0.648  | 0.487 | 0.808   |
| Logistic Reg + SMOTE        | 0.828    | 0.356     | 0.657  | 0.462 | 0.801   |
| Random Forest + SMOTE       | 0.890    | 0.515     | 0.371  | 0.431 | 0.778   |
| **Tuned GB (GridSearchCV)** | **0.902** | **0.662** | **0.270** | **0.384** | **0.814** |

### Cross-Validation Results (5-Fold Stratified, ROC-AUC)

| Model                    | CV ROC-AUC (Mean ± Std)  |
|--------------------------|--------------------------|
| Gradient Boosting        | 0.7932 ± 0.0032          |
| Logistic Regression      | 0.7889 ± 0.0066          |
| XGBoost (weighted)       | 0.7863 ± 0.0037          |
| Random Forest (balanced) | 0.7696 ± 0.0083          |

### Best Hyperparameters (GridSearchCV)
- `learning_rate`: 0.05
- `max_depth`: 5
- `n_estimators`: 200
- `subsample`: 1.0
- **Best CV ROC-AUC**: 0.7967

The **Tuned Gradient Boosting** model (via GridSearchCV over n_estimators, learning_rate, max_depth, subsample) achieved the best overall ROC-AUC (0.814) on the test set.

---

## 5. Feature Importance

The top predictive features from the tuned Gradient Boosting model are:

1. **`nr.employed`** — Number of employees (quarterly indicator). The strongest predictor, reflecting broader economic conditions.
2. **`euribor3m`** — 3-month Euribor rate. Low rates incentivize term deposits as alternative savings vehicles.
3. **`emp.var.rate`** — Employment variation rate. Correlated with economic uncertainty.
4. **`cons.conf.idx`** — Consumer confidence index.
5. **`age`** — Client age.
6. **`campaign`** — Number of contacts during this campaign.
7. **`pdays`** — Days since previous contact (999 = never contacted).
8. **`cons.price.idx`** — Consumer price index.
9. **`poutcome_success`** — Whether the previous campaign outcome was successful.
10. **`contact_cellular`** — Whether the contact was via cellular.

**Key insight**: Macroeconomic features dominate the top of the feature importance ranking — suggesting that external economic conditions are more predictive of subscription than individual client demographics. This has practical implications: campaign timing (launching during favorable economic windows) may matter more than client targeting.

---

## 6. Error Analysis

### Misclassification Breakdown
Using the tuned Gradient Boosting model on the test set (928 actual subscribers, 7,310 non-subscribers):
- **False Negatives** (missed subscribers): **677** — the model misses 73% of actual subscribers. These clients tend to have predicted probabilities clustered just below the 0.5 decision threshold.
- **False Positives** (wasted calls): **128** — fewer in number but represent wasted campaign resources.

### Classification Report (Tuned GB)

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| No        | 0.91      | 0.98   | 0.95     | 7,310   |
| Yes       | 0.66      | 0.27   | 0.38     | 928     |
| **Accuracy** |        |        | **0.90** | 8,238   |
| Macro Avg | 0.79      | 0.63   | 0.67     | 8,238   |
| Weighted Avg | 0.89   | 0.90   | 0.88     | 8,238   |

### False Negative Profile
The 677 missed subscribers have a mean age of ~40, mean `emp.var.rate` of -1.03, and mean `euribor3m` of 2.46 — broadly similar to the overall subscriber population, making them inherently hard to distinguish from non-subscribers.

### Precision-Recall Tradeoff
The precision-recall curve reveals a steep tradeoff: achieving high recall (>70%) requires accepting precision below 25%. The current 0.5 threshold favors precision over recall.

**Business recommendation**: Since the cost of a phone call is low relative to the revenue from a new term deposit, the optimal threshold should be **lowered below 0.5** to capture more potential subscribers. A threshold of ~0.3 could roughly double recall while maintaining acceptable precision — the exact threshold should be tuned based on the bank's cost structure (cost per call vs. expected revenue per subscription).

### Characteristics of Misclassified Clients
- **False negatives** are disproportionately admin and blue-collar workers aged 30–50 — the "hardest to predict" demographic where individual variation is highest.
- **False positives** often have characteristics similar to true subscribers (e.g., contacted via cellular, favorable economic indicators) but ultimately did not subscribe — potentially clients who were interested but not converted.

---

## Summary

The tuned Gradient Boosting classifier achieves the best discriminative performance (ROC-AUC 0.814) on this dataset after excluding the `duration` feature. Macroeconomic conditions are the dominant predictors, suggesting that **campaign timing** is at least as important as **client selection**. Class imbalance was addressed through class weighting, SMOTE, and XGBoost's `scale_pos_weight` — with XGBoost (weighted) achieving the best F1 score (0.487) due to its higher recall (0.648), while the tuned Gradient Boosting achieved the best ROC-AUC (0.814). For deployment, we recommend threshold optimization based on the bank's specific cost-benefit structure to balance missed opportunities against wasted outreach.
