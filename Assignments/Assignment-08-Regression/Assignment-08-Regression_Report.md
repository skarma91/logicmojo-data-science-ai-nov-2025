# Assignment 8 — House Prices Regression: Project Report

## 1. Problem Framing & Evaluation Metrics

**Goal:** Predict the sale price of residential homes in Ames, Iowa using 79 explanatory features (36 numeric, 43 categorical).

**Primary metric — RMSE:** Expressed in USD, directly interpretable. It penalises large errors more than MAE, which is desirable when a single large miss is costlier than many small ones.

**Secondary metric — RMSLE:** The target is right-skewed (skew = 1.88); RMSLE provides a scale-invariant view that weights percentage errors equally across the price range.

---

## 2. Key EDA Insights

### Target Distribution
- SalePrice ranges from ~$34 k to ~$755 k with a median of ~$163 k.
- The distribution is right-skewed (skew = 1.88, kurtosis = 6.54). Applying `log1p` yields a near-normal distribution (skew = 0.12).

### Missing Values
- 19 features contain missing values. Five columns (PoolQC, MiscFeature, Alley, Fence, FireplaceQu) have >40 % missing; in every case NA indicates the feature is absent (no pool, no alley, etc.), so categorical imputation with a "Missing" label is semantically correct.
- `LotFrontage` (~18 % missing) is numeric and imputed with the median.

### Top Predictors (Pearson Correlation with SalePrice)
| Feature | r |
|---|---|
| OverallQual | 0.79 |
| GrLivArea | 0.71 |
| GarageCars | 0.64 |
| GarageArea | 0.62 |
| TotalBsmtSF | 0.61 |
| 1stFlrSF | 0.61 |
| FullBath | 0.56 |
| YearBuilt | 0.52 |

- Scatterplots confirm strong monotonic relationships for OverallQual and GrLivArea.
- Box plots reveal that neighborhoods NoRidge, NridgHt, and StoneBr command the highest median prices.
- A few outliers (GrLivArea > 4000 sf with SalePrice < $200 k) are visible but retained to avoid data-snooping.

---

## 3. Feature Engineering Summary

Nine engineered features were added on top of the original 79:

| Feature | Definition |
|---|---|
| TotalSF | TotalBsmtSF + 1stFlrSF + 2ndFlrSF |
| TotalBath | FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath |
| TotalPorchSF | Sum of all porch areas |
| HouseAge | YrSold − YearBuilt |
| RemodAge | YrSold − YearRemodAdd |
| HasGarage | Binary: GarageArea > 0 |
| HasBsmt | Binary: TotalBsmtSF > 0 |
| Has2ndFlr | Binary: 2ndFlrSF > 0 |
| HasPool | Binary: PoolArea > 0 |

After one-hot encoding, the engineered feature set expanded to 310 transformed columns. Lasso regularisation (alpha = 50) retained 149 of them, confirming that roughly half the one-hot indicators are redundant.

---

## 4. Model Comparison

All models share the same 80/20 train-test split (random_state = 42). Cross-validated RMSE is 5-fold on the training set. Models marked "+ FE" use the engineered feature set.

| Model | Train RMSE | Test RMSE | CV RMSE | Test RMSLE |
|---|---|---|---|---|
| Linear Regression | 18,960 | *unstable* | *unstable* | — |
| Ridge (alpha=10) | 24,359 | 30,521 | 32,637 | — |
| Lasso (alpha=100) | 22,001 | 28,225 | 33,052 | — |
| Decision Tree (depth=6) | 23,014 | 42,126 | 44,335 | — |
| Random Forest | 11,129 | 29,229 | 30,625 | — |
| Gradient Boosting | 4,661 | 26,573 | 29,933 | — |
| XGBoost | 5,431 | 25,307 | 29,017 | — |
| Ridge + FE | 23,888 | 30,080 | 32,575 | — |
| Random Forest + FE | 10,813 | 29,187 | 29,583 | — |
| Gradient Boosting + FE | 4,304 | 27,819 | 27,784 | — |
| XGBoost + FE | 5,366 | 25,510 | 29,011 | — |
| **XGBoost Tuned + FE** | **5,446** | **24,755** | **26,897** | — |

**Observations:**
- Plain Linear Regression is numerically unstable due to multicollinearity in the one-hot-encoded space; Ridge and Lasso stabilise it.
- Tree ensembles (Random Forest, Gradient Boosting, XGBoost) comfortably outperform linear baselines.
- Feature engineering improved CV RMSE for Gradient Boosting (29,933 → 27,784) and Random Forest (30,625 → 29,583).
- Hyperparameter tuning of XGBoost (500 trees, lr = 0.05, max_depth = 4, subsample = 0.8) yielded the best overall test RMSE of **$24,755** and CV RMSE of **$26,897**.

---

## 5. Model Interpretation — Feature Importance

### XGBoost (Best Model) — Top Features by Gain
The tuned XGBoost model assigns the most importance to:
1. **OverallQual** — overall material and finish quality (dominant predictor).
2. **GrLivArea / TotalSF** — above-grade and total living area.
3. **TotalBath** — combined bathroom count.
4. **GarageCars / GarageArea** — garage size.
5. **Neighborhood** indicators (NridgHt, NoRidge, StoneBr contribute positively).
6. **YearBuilt / HouseAge** — newer homes command higher prices.

### Ridge Coefficients (Linear Interpretation)
The standardised Ridge coefficients confirm that OverallQual, GrLivArea, and Neighborhood_NridgHt are the largest positive drivers, while features like MSSubClass (numerically coded dwelling type) and OverallCond have smaller, sometimes negative, effects — reinforcing that raw condition scores contain noise relative to quality scores.

---

## 6. Error Analysis

### Residual Patterns
- Residuals are approximately centred at zero with a slight positive tail (under-prediction of expensive homes).
- Heteroscedasticity is visible: variance increases with predicted price, a common pattern in housing data.

### Worst Predictions
The 10 largest absolute errors share common traits:
- **Unusual sale conditions** (e.g., partial sales, family transactions) that deviate from market norms.
- **Extreme living areas** coupled with moderate quality ratings.
- **Older homes** in premium neighborhoods that received major remodels — the model under-estimates the value added by renovation.

### Improvement Suggestions
1. **Outlier treatment:** Clip or remove the 2–3 homes with GrLivArea > 4000 sf and SalePrice < $200 k (likely partial demolitions or data-entry errors).
2. **Target encoding:** Replace one-hot encoding of high-cardinality categoricals (Neighborhood, Exterior) with target-mean encoding to reduce dimensionality and capture ordinal price gradients.
3. **Ordinal encoding:** Quality features (ExterQual, BsmtQual, KitchenQual) have a natural order (Po < Fa < TA < Gd < Ex) — ordinal encoding preserves this and reduces feature count.
4. **Model stacking:** Blend the best tree-based model with a regularised linear model to capture both non-linear interactions and stable linear trends.
5. **Log-target training:** Train models on log(SalePrice) and back-transform predictions to directly optimise RMSLE and mitigate heteroscedasticity.
