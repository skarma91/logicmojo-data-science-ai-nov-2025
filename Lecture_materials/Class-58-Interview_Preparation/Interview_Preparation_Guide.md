# Data Science, Machine Learning & AI — Interview Preparation Guide

*LogicMojo Data Science & AI — November 2025 Batch*

A topic-wise reference for preparing for **Data Scientist / ML** and **GenAI / LLM / AI Engineer** interviews. It mirrors our course curriculum, grouped into consolidated interview topic areas. Each section lists the concepts to master, what interviewers tend to probe, a few **representative interview questions with answer signals**, and a curated set of high-quality external resources. Your own lecture notebooks and notes in the [course repository](https://github.com/skarma91/logicmojo-data-science-ai-nov-2025) remain your primary, fastest reference — the *Course material* line in each section points you to the relevant class folder.

> **How to read the sample questions:** each comes with an *answer signal* — a one-line summary of what a strong answer must cover, **not** a full model answer. Try to answer out loud first, then check whether you hit the signal. If you can't, that's your cue to study the linked resources.

---

## How to use this guide

- **Don't read it linearly.** Use the topic index to jump to whatever a given interview emphasizes.
- **Concept first, resource second.** Skim the "Master these concepts" list and self-rate. Spend time only where you're shaky, then use the linked resources to close the gap.
- **Re-derive, don't re-read.** For interviews, the test is whether you can explain a concept from scratch on a whiteboard and connect it to trade-offs. Practice explaining each bullet out loud.
- **Pair every algorithm with: intuition → math → assumptions → when it fails → how to evaluate.** That five-part frame answers ~80% of ML interview follow-ups.
- **Build a project narrative.** Most rounds end with "tell me about a project." Have 2–3 projects (ideally from the capstones) you can discuss end-to-end: problem framing, data, modeling choices, evaluation, deployment, and what you'd do differently.

---

## What these interviews look like

**Data Scientist / ML roles** typically include some mix of:
1. **Statistics & probability** — conceptual reasoning, experimentation/A-B testing.
2. **ML breadth & depth** — algorithms, bias–variance, evaluation, feature engineering.
3. **Coding** — Python + (very often) SQL; sometimes DSA.
4. **ML case / system design** — "design a model for X," metrics, data, iteration.
5. **Behavioral & project deep-dive.**

**GenAI / LLM / AI Engineer roles** add:
1. **Deep learning & transformers** — architecture, attention, training/fine-tuning.
2. **LLM application design** — prompting, RAG, agents, evaluation, guardrails, cost/latency.
3. **Practical engineering** — embeddings, vector stores, orchestration frameworks, deployment.

A realistic prep arc: **foundations (Python, math, stats)** → **classical ML** → **deep learning** → **NLP & transformers** → **GenAI (RAG/agents)** → **system design + behavioral**, with coding/SQL practice running in parallel throughout.

---

## Topic index

| # | Topic area | Course material |
|---|------------|-----------------|
| 1 | Programming & Python for Data Science | Classes 1–9 |
| 2 | Mathematics for ML | Classes 10–13 |
| 3 | Probability | Classes 14–16 |
| 4 | Statistics & Inference (incl. A/B testing) | Classes 17–21 |
| 5 | ML Foundations & Workflow | Class 22 |
| 6 | Regression | Classes 23–24 |
| 7 | Classification Algorithms | Classes 25–26, 29–32 |
| 8 | Bias–Variance, Regularization & Model Evaluation | Classes 27–28, 33 |
| 9 | Ensemble Methods & Boosting | Classes 34–35 |
| 10 | Dimensionality Reduction & Unsupervised Learning | Classes 36, 38–39 |
| 11 | Feature Engineering & Imbalanced Data | Class 37 |
| 12 | Time Series Analysis & Forecasting | Class 49 |
| 13 | Model Deployment & MLOps | Class 40 |
| 14 | Deep Learning Foundations | Classes 41–44 |
| 15 | Computer Vision & CNNs | Classes 45–46 |
| 16 | NLP Foundations | Classes 47–48 |
| 17 | Sequence Models (RNN/LSTM/Attention) | Classes 50–51 |
| 18 | Transformers & LLMs | Classes 52–53 |
| 19 | Generative AI: Prompting, RAG & Agents | Classes 54–57 |
| A | ML & GenAI System Design (General Skills) | — |
| B | SQL, Coding & DSA (General Skills) | — |
| C | Behavioral, Projects & Communication (General Skills)| — |
| D | General resource hub & practice platforms | — |

---

# Part I — Foundations

## 1. Programming & Python for Data Science
*Course material: Classes 1–9 (Python basics → advanced → NumPy / Pandas / visualization)*

**Master these concepts**
- Core Python: data types, control flow, functions, comprehensions, generators, decorators, OOP, exception handling, scope/closures.
- Complexity basics: time/space complexity, mutability, when to use list vs. set vs. dict.
- NumPy: vectorization, broadcasting, slicing, why vectorized ops beat Python loops.
- Pandas: indexing/`loc`/`iloc`, `groupby`-aggregate-transform, joins/merges, reshaping (`pivot`/`melt`), handling missing data, method chaining.
- Visualization: choosing the right chart; Matplotlib/Seaborn fundamentals; reading a plot critically.

**What interviewers probe:** clean, idiomatic Python under time pressure; manipulating a DataFrame to answer a question; explaining *why* vectorization matters.

**Sample interview questions**
1. **List vs. tuple vs. set vs. dict — when would you use each?** — *Signal:* mutability, ordering, uniqueness, and O(1) membership/lookup via hashing.
2. **Why is NumPy vectorization faster than a Python `for` loop?** — *Signal:* contiguous typed memory + C-level loops/SIMD; avoids per-element Python object overhead.
3. **What is a generator and when is it preferable to a list?** — *Signal:* lazy, single-pass evaluation; memory-efficient for large/streaming data.
4. **In Pandas, how do `apply` and vectorized operations differ in performance?** — *Signal:* prefer built-in vectorized/`str`/`dt` ops; row-wise `apply` runs Python per row and is slow.

**Core resources**
- [Python official tutorial](https://docs.python.org/3/tutorial/) — authoritative language reference.
- [Pandas — Getting started & user guide](https://pandas.pydata.org/docs/) — the canonical docs; work through "10 minutes to pandas."
- [NumPy — the absolute basics](https://numpy.org/doc/stable/user/absolute_beginners.html) and [broadcasting docs](https://numpy.org/doc/stable/user/basics.broadcasting.html).
- [Kaggle Learn: Python, Pandas, Data Visualization](https://www.kaggle.com/learn) — short, hands-on micro-courses.

---

## 2. Mathematics for Machine Learning
*Course material: Classes 10–13 (linear algebra, calculus, optimization)*

**Master these concepts**
- **Linear algebra:** vectors/matrices, matrix multiplication, rank, inverse, determinant, eigenvalues/eigenvectors, orthogonality, projections, SVD — and their ML meaning (e.g., why eigen-decomposition underlies PCA).
- **Calculus:** derivatives, partial derivatives, gradients, chain rule (the backbone of backpropagation), Jacobians/Hessians at a conceptual level.
- **Optimization:** convexity, gradient descent and variants (batch/stochastic/mini-batch), learning rate, local vs. global minima, the role of the loss surface.

**What interviewers probe:** geometric intuition (not just formulas) — "what does an eigenvector *mean* here?", "why does gradient descent move that direction?"

**Sample interview questions**
1. **What is an eigenvector/eigenvalue, and where does it appear in ML?** — *Signal:* direction unchanged under a transform, scaled by the eigenvalue; PCA uses covariance eigenvectors.
2. **Why does gradient descent step in the negative-gradient direction?** — *Signal:* gradient points toward steepest ascent of the loss; the negative is steepest local descent.
3. **What does convexity guarantee for optimization?** — *Signal:* every local minimum is global → reliable convergence (e.g., linear/logistic regression losses).
4. **What is SVD and why is it useful?** — *Signal:* factorization enabling low-rank approximation/dimensionality reduction; underlies PCA.

**Core resources**
- [3Blue1Brown — *Essence of Linear Algebra*](https://www.3blue1brown.com/topics/linear-algebra) and [*Essence of Calculus*](https://www.3blue1brown.com/topics/calculus) — the best intuition-building visual series available.
- [*Mathematics for Machine Learning* (Deisenroth, Faisal, Ong)](https://mml-book.github.io/) — free textbook tying the math directly to ML.
- [Khan Academy — Linear Algebra](https://www.khanacademy.org/math/linear-algebra) — for filling specific gaps with practice.

---

## 3. Probability
*Course material: Classes 14–16*

**Master these concepts**
- Sample spaces, axioms, conditional probability, independence, the law of total probability, **Bayes' theorem** (expect to apply it).
- Random variables; common distributions (Bernoulli, Binomial, Poisson, Uniform, Normal, Exponential) and *when each arises*.
- Expectation, variance, covariance, correlation; properties of expectation/variance.
- Joint/marginal/conditional distributions; the **Central Limit Theorem** and **Law of Large Numbers** and why they matter for inference.

**What interviewers probe:** Bayes-rule word problems, distribution selection, expected-value reasoning, and connecting probability to model assumptions (e.g., Naive Bayes, MLE).

**Sample interview questions**
1. **A test is 99% accurate for a disease with 0.1% prevalence; you test positive — what's the chance you actually have it?** — *Signal:* apply Bayes' theorem; the low base rate dominates, so the answer is surprisingly low (~9%) — the base-rate fallacy.
2. **When would you model counts as Poisson vs. Binomial?** — *Signal:* Binomial = fixed n trials with probability p; Poisson = count of rare events per interval (limit of Binomial).
3. **Covariance vs. correlation — what's the difference?** — *Signal:* correlation is scale-normalized covariance bounded in [−1, 1], comparable across variables.
4. **State the Central Limit Theorem and why it matters.** — *Signal:* sample means tend to normal for large n regardless of population shape (finite variance); justifies confidence intervals and many tests.

**Core resources**
- [Harvard Stat 110 — *Introduction to Probability* (Blitzstein & Hwang)](https://projects.iq.harvard.edu/stat110/home) — free lectures, problem sets, and the book; the gold standard for probabilistic reasoning.
- [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/) — interactive visual intro to probability and statistics.
- [StatQuest — Probability & distributions playlists](https://www.youtube.com/@statquest) — clear, exam-oriented explanations.

---

## 4. Statistics & Inference (including A/B testing)
*Course material: Classes 17–21 (descriptive & inferential statistics)*

**Master these concepts**
- **Descriptive:** measures of central tendency/dispersion, skew/kurtosis, percentiles, correlation vs. causation.
- **Sampling & estimation:** sampling distributions, standard error, point vs. interval estimates, **confidence intervals**, bias/consistency of estimators, **MLE**.
- **Hypothesis testing:** null/alternative, p-value (and its correct interpretation), significance level, **Type I/II errors**, statistical power, t-tests, z-tests, chi-square, ANOVA; one- vs. two-tailed.
- **Experimentation / A/B testing:** randomization, sample-size/power calculation, multiple-comparisons problem, peeking/early-stopping pitfalls, choosing a metric, practical vs. statistical significance.

**What interviewers probe:** *correct* p-value/CI interpretation (a very common trap), designing and analyzing an A/B test end-to-end, and reasoning about confounders.

**Sample interview questions**
1. **What does a p-value actually mean?** — *Signal:* P(data this extreme or more | H₀ true); it is **not** P(H₀ true) and not an effect size.
2. **Type I vs. Type II error — and the trade-off.** — *Signal:* false positive vs. false negative; lowering α raises β unless you increase sample size/power.
3. **Design an A/B test for a new checkout button.** — *Signal:* hypothesis + primary metric, randomization unit, power/sample size computed up front, run full business cycles, guardrail metrics, avoid peeking, judge practical (not just statistical) significance.
4. **Your test is "significant" after one day — do you ship?** — *Signal:* no; peeking inflates false positives, predetermine duration/sample size, watch novelty effects and weekly seasonality.

**Core resources**
- [*An Introduction to Statistical Learning* (ISL) — free book + slides](https://www.statlearning.com/) — chapters on inference and resampling are interview-grade.
- [StatQuest — Statistics Fundamentals](https://www.youtube.com/@statquest) — hypothesis testing, p-values, CIs explained intuitively.
- [Trustworthy Online Controlled Experiments (Kohavi, Tang, Xu)](https://experimentguide.com/) — the standard reference for A/B testing; companion site to the book.
- [Seeing Theory — Frequentist Inference](https://seeing-theory.brown.edu/frequentist-inference/index.html) — visual treatment of CIs and testing.

---

# Part II — Classical Machine Learning

## 5. ML Foundations & Workflow
*Course material: Class 22 (introduction to machine learning)*

**Master these concepts**
- Supervised vs. unsupervised vs. (briefly) reinforcement learning; regression vs. classification.
- The end-to-end ML workflow: problem framing → data → features → model → evaluation → deployment → monitoring.
- Train/validation/test splits, cross-validation, data leakage (a top interview topic), overfitting vs. underfitting at a high level.
- Parametric vs. non-parametric models; the **no-free-lunch** idea and the bias–variance trade-off as a framing device.

**What interviewers probe:** can you structure a vague problem ("predict churn") into a concrete ML task with the right target, data, and metric?

**Sample interview questions**
1. **What is data leakage? Give an example.** — *Signal:* train-time access to test/future/target-derived info (e.g., scaling before the split); inflates offline metrics, fails in production.
2. **When do you use a simple holdout vs. k-fold cross-validation?** — *Signal:* holdout for large data; k-fold for small data/variance reduction; stratify for imbalance; time-aware splits for temporal data.
3. **Turn "reduce customer churn" into an ML problem.** — *Signal:* define churn and target horizon, the prediction unit, label window, candidate features, and a metric tied to the business action.

**Core resources**
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) — concise, well-structured foundations with exercises.
- [ISL, Ch. 2 — Statistical Learning](https://www.statlearning.com/) — the cleanest treatment of the core framing.
- [*Hands-On Machine Learning* (Géron) — code repo](https://github.com/ageron/handson-ml3) — practical companion notebooks for the whole classical-ML stack.

---

## 6. Regression
*Course material: Classes 23–24 (simple & multiple linear regression, gradient descent)*

**Master these concepts**
- Simple & multiple linear regression; the normal equation vs. gradient-descent solutions.
- **Assumptions** (linearity, independence, homoscedasticity, normality of residuals, no multicollinearity) — and how to check/violate them.
- Cost function (MSE), residual analysis, R²/adjusted R², interpreting coefficients.
- Polynomial features and the move from linear to non-linear fits; multicollinearity and its effects.

**What interviewers probe:** stating and checking assumptions, deriving the gradient-descent update, and interpreting coefficients correctly (including the effect of correlated features).

**Sample interview questions**
1. **List linear regression's assumptions and what breaks if they're violated.** — *Signal:* linearity, independence, homoscedasticity, normal residuals, low multicollinearity; violations harm coefficient estimates and inference.
2. **Normal equation vs. gradient descent — when use which?** — *Signal:* normal equation is closed-form but O(p³) and needs invertible XᵀX; gradient descent scales to large p/n and supports online learning.
3. **How do you interpret a coefficient, and what does multicollinearity do to it?** — *Signal:* expected change in y per unit x with others held fixed; collinearity inflates coefficient variance → unstable/uninterpretable (check VIF).

**Core resources**
- [ISL, Ch. 3 — Linear Regression](https://www.statlearning.com/) — assumptions, inference, and interpretation done rigorously.
- [StatQuest — Linear Regression and Gradient Descent](https://www.youtube.com/@statquest).
- [scikit-learn — Linear Models user guide](https://scikit-learn.org/stable/modules/linear_model.html) — practical API plus the underlying math notes.

---

## 7. Classification Algorithms
*Course material: Classes 25–26 (logistic & multi-class), 29 (kNN, SVM), 30 (Bayes), 31 (decision trees), 32 (random forest)*

**Master these concepts**
- **Logistic regression:** sigmoid, log-odds, cross-entropy loss, decision boundary; why it's linear in feature space.
- **Multi-class:** one-vs-rest vs. softmax/multinomial.
- **kNN:** distance metrics, choice of *k*, curse of dimensionality, lazy learning.
- **SVM:** margins, support vectors, the kernel trick, soft margin / C, common kernels.
- **Naive Bayes:** the conditional-independence assumption, why it works despite being "naive," variants (Gaussian/Multinomial/Bernoulli).
- **Decision trees:** splitting criteria (Gini, entropy/information gain), pruning, overfitting tendency, interpretability.
- **Random forest:** bagging, feature subsampling, why ensembling reduces variance, out-of-bag error, feature importance.

**What interviewers probe:** comparing algorithms ("logistic regression vs. SVM vs. random forest for this problem — why?"), the kernel trick, and the assumptions/failure modes of each.

**Sample interview questions**
1. **Why is logistic regression a linear classifier despite the sigmoid?** — *Signal:* the decision boundary (log-odds = 0) is linear in the features; the sigmoid only maps the score to a probability.
2. **Explain the kernel trick.** — *Signal:* compute inner products in a high-dimensional space via a kernel without ever mapping explicitly → non-linear boundaries (RBF, polynomial).
3. **Why is Naive Bayes "naive," and why does it still work?** — *Signal:* assumes features conditionally independent given the class; often false, but the resulting boundary is still good, fast, and robust for high-dim/text.
4. **What does a random forest fix relative to a single decision tree?** — *Signal:* a single tree is high-variance/overfits; bagging + feature subsampling decorrelate trees and averaging reduces variance.

**Core resources**
- [StatQuest — full ML playlist](https://www.youtube.com/@statquest) (logistic regression, SVM, trees, random forests — each explained step by step).
- [ISL, Ch. 4, 8, 9](https://www.statlearning.com/) — classification, tree-based methods, SVMs.
- [scikit-learn — Supervised learning user guide](https://scikit-learn.org/stable/supervised_learning.html) — concise theory + practical guidance per algorithm.

---

## 8. Bias–Variance, Regularization & Model Evaluation
*Course material: Classes 27–28 (bias–variance, regularization), 33 (metrics & hyperparameter optimization)*

**Master these concepts**
- **Bias–variance decomposition**; how model complexity, data size, and noise move each term.
- **Regularization:** L1 (Lasso) vs. L2 (Ridge) — geometric intuition, effect on weights, feature selection via L1, Elastic Net.
- **Classification metrics:** confusion matrix, precision, recall, F1, ROC–AUC vs. PR–AUC, when accuracy is misleading, threshold selection.
- **Regression metrics:** MAE, MSE/RMSE, R²; when to prefer each.
- **Hyperparameter tuning:** grid vs. random search, Bayesian optimization (concept), cross-validation strategies, the train/validation/test discipline.

**What interviewers probe:** L1 vs. L2 (expect the diamond-vs-circle intuition), choosing the right metric for an imbalanced or cost-sensitive problem, and avoiding leakage during tuning.

**Sample interview questions**
1. **Explain the bias–variance trade-off.** — *Signal:* simple models → high bias/low variance (underfit), complex → low bias/high variance (overfit); total error = bias² + variance + irreducible noise.
2. **L1 vs. L2 regularization — difference and when to use each.** — *Signal:* L1/Lasso produces sparsity/feature selection (diamond corners); L2/Ridge shrinks smoothly and handles collinearity; Elastic Net combines both.
3. **When is accuracy a poor metric, and what do you use instead?** — *Signal:* under class imbalance; use precision/recall/F1, PR-AUC, ROC-AUC, chosen by the relative cost of false positives vs. false negatives.
4. **ROC-AUC vs. PR-AUC — when prefer PR?** — *Signal:* PR-AUC is more informative when positives are rare and are the class of interest.

**Core resources**
- [ISL, Ch. 5 (Resampling) & Ch. 6 (Regularization/Shrinkage)](https://www.statlearning.com/).
- [Google ML Crash Course — Classification & metrics modules](https://developers.google.com/machine-learning/crash-course) — clean treatment of ROC/AUC, precision/recall.
- [scikit-learn — Model evaluation & metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) and [tuning hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html).

---

## 9. Ensemble Methods & Boosting
*Course material: Classes 34 (AdaBoost), 35 (Gradient Boosting)*

**Master these concepts**
- Bagging vs. boosting vs. stacking — and *why* each helps (variance vs. bias).
- **AdaBoost:** reweighting misclassified points, weak learners, additive modeling.
- **Gradient boosting:** fitting to residuals/gradients, learning rate/shrinkage, number of trees, tree depth.
- **XGBoost / LightGBM / CatBoost** (highly common in practice): regularization, handling missing values, why they dominate tabular competitions; key hyperparameters.

**What interviewers probe:** bagging vs. boosting trade-offs, how gradient boosting actually fits residuals, and tuning/regularizing a boosted model to avoid overfitting.

**Sample interview questions**
1. **Bagging vs. boosting — what error does each target?** — *Signal:* bagging reduces variance (parallel, independent learners); boosting reduces bias (sequential learners fitting prior errors).
2. **How does gradient boosting work?** — *Signal:* an additive model where each new tree fits the negative gradient/residuals of the loss, scaled by a learning rate.
3. **Why does XGBoost usually win on tabular data, and which knobs prevent overfitting?** — *Signal:* regularized boosting + missing-value handling + speed; lower learning rate with more trees, plus max_depth, subsample, min_child_weight, reg_lambda.

**Core resources**
- [StatQuest — AdaBoost, Gradient Boost, and XGBoost series](https://www.youtube.com/@statquest) — the clearest walkthroughs of the math.
- [XGBoost — official documentation & "Introduction to Boosted Trees"](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).
- [ISL, Ch. 8 — Tree-based methods (boosting section)](https://www.statlearning.com/).

---

## 10. Dimensionality Reduction & Unsupervised Learning
*Course material: Classes 36 (PCA & dimensionality reduction), 38–39 (clustering)*

**Master these concepts**
- **PCA:** variance maximization, covariance matrix, eigenvectors/eigenvalues, explained variance, scaling before PCA, what the components mean; relationship to SVD.
- Other reductions (conceptual): t-SNE/UMAP for visualization (and their limits).
- **Clustering:** k-means (objective, initialization, choosing *k*, elbow/silhouette), hierarchical clustering (linkages, dendrograms), DBSCAN (density, handling noise/arbitrary shapes).
- Evaluating clusters without labels; the curse of dimensionality.

**What interviewers probe:** the mechanics and assumptions of PCA, choosing *k* defensibly, and k-means vs. DBSCAN trade-offs.

**Sample interview questions**
1. **Walk through PCA — and why scale features first?** — *Signal:* center → covariance → eigen/SVD → keep top components by explained variance; scaling stops high-variance units from dominating.
2. **How do you choose k in k-means?** — *Signal:* elbow on inertia, silhouette score, and domain knowledge; remember k-means assumes roughly spherical, equal-size clusters.
3. **k-means vs. DBSCAN — when use which?** — *Signal:* k-means needs k and finds spherical clusters; DBSCAN finds arbitrary shapes and labels noise/outliers but needs eps/minPts.

**Core resources**
- [StatQuest — PCA (step-by-step) and Clustering](https://www.youtube.com/@statquest).
- [ISL, Ch. 12 — Unsupervised Learning](https://www.statlearning.com/).
- [scikit-learn — Clustering](https://scikit-learn.org/stable/modules/clustering.html) and [Decomposition/PCA](https://scikit-learn.org/stable/modules/decomposition.html) guides (with comparison tables).

---

## 11. Feature Engineering & Imbalanced Data
*Course material: Class 37*

**Master these concepts**
- Encoding categoricals (one-hot, ordinal, target/mean encoding and its leakage risk), scaling/normalization, binning, interaction features, date/time features.
- Handling missing data (deletion vs. imputation strategies) and outliers.
- Feature selection (filter/wrapper/embedded methods) vs. dimensionality reduction.
- **Imbalanced classification:** resampling (SMOTE, under/over-sampling), class weights, threshold moving, and *choosing the right metric* (PR-AUC, recall) instead of accuracy.

**What interviewers probe:** preventing leakage in feature engineering and the full toolkit for a heavily imbalanced target (e.g., fraud).

**Sample interview questions**
1. **How do you encode a high-cardinality categorical without leakage?** — *Signal:* out-of-fold/CV target encoding (or smoothing), hashing, or learned embeddings; never fit the encoder on the full dataset before splitting.
2. **A fraud dataset is 99.5% negatives — how do you handle it?** — *Signal:* pick the right metric (PR-AUC/recall), use class weights, resample (SMOTE/undersample) *inside* CV folds, tune the threshold, consider anomaly framing.
3. **When does feature scaling matter and when doesn't it?** — *Signal:* matters for distance/gradient-based models (kNN, SVM, NN, regularized linear, PCA); not for tree-based models.

**Core resources**
- [Kaggle Learn — Feature Engineering](https://www.kaggle.com/learn/feature-engineering) — practical, hands-on.
- [*Feature Engineering for Machine Learning* (Zheng & Casari)](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — a focused reference (O'Reilly).
- [imbalanced-learn (imblearn) — user guide](https://imbalanced-learn.org/stable/user_guide.html) — SMOTE and resampling techniques with theory.

---

## 12. Time Series Analysis & Forecasting
*Course material: Class 49*

**Master these concepts**
- Components: trend, seasonality, cyclicity, noise; stationarity and how to test/achieve it (differencing).
- Autocorrelation (ACF/PACF); classical models: AR, MA, **ARIMA/SARIMA**, exponential smoothing.
- Train/test splitting for time series (no shuffling!), rolling/expanding windows, backtesting.
- Forecast evaluation (MAE/RMSE/MAPE); leakage specific to temporal data.

**What interviewers probe:** stationarity and why it matters, choosing/interpreting ARIMA terms, and *correct* validation for temporal data.

**Sample interview questions**
1. **What is stationarity and why does it matter?** — *Signal:* constant mean/variance/autocorrelation over time; assumed by models like ARIMA; achieved via differencing; test with ADF.
2. **How do you validate a forecasting model?** — *Signal:* never shuffle; use chronological splits and rolling/expanding-window backtests; report MAE/RMSE/MAPE.
3. **What do p, d, q mean in ARIMA?** — *Signal:* AR order, differencing order, MA order; inferred from PACF/ACF and the differencing needed for stationarity.

**Core resources**
- [*Forecasting: Principles and Practice* (Hyndman & Athanasopoulos)](https://otexts.com/fpp3/) — the free, authoritative forecasting textbook.
- [StatQuest / time-series tutorials](https://www.youtube.com/@statquest) — for ACF/PACF and ARIMA intuition.

---

## 13. Model Deployment & MLOps
*Course material: Class 40 (model deployment using Streamlit)*

**Master these concepts**
- Serialization (pickle/joblib/ONNX), building an inference API, batch vs. real-time serving.
- Lightweight apps with Streamlit; the difference between a demo and a production service.
- The MLOps lifecycle: versioning (data/model/code), CI/CD for ML, monitoring, **data/concept drift**, retraining triggers, reproducibility.
- Training–serving skew, latency/throughput/cost trade-offs.

**What interviewers probe (esp. MLE roles):** how you'd take a notebook model to production, what you'd monitor, and how you'd detect/handle drift.

**Sample interview questions**
1. **How would you take a notebook model to production?** — *Signal:* serialize → wrap in an API/service → containerize → choose batch vs. real-time → tests + CI/CD → monitoring and rollback.
2. **What are data drift and concept drift, and how do you detect them?** — *Signal:* input-distribution shift vs. P(y|x) shift; monitor feature/label distributions and metrics, run statistical tests, trigger retraining.
3. **What is training–serving skew?** — *Signal:* mismatch between training and serving feature pipelines/data; mitigate with shared transformations or a feature store.

**Core resources**
- [Streamlit — documentation & gallery](https://docs.streamlit.io/) — for the deployment piece in our course.
- [Made With ML (Goku Mohandas)](https://madewithml.com/) — free, end-to-end MLOps course (design → develop → deploy → iterate).
- [*Designing Machine Learning Systems* (Chip Huyen)](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html) and her [MLOps resource hub](https://huyenchip.com/mlops/) — the most interview-relevant production-ML reading.

---

# Part III — Deep Learning

## 14. Deep Learning Foundations
*Course material: Classes 41–44 (intro to ANN & PyTorch, training neural networks)*

**Master these concepts**
- Perceptron → MLP; forward pass; **backpropagation** (be ready to explain it via the chain rule).
- Activations (sigmoid, tanh, ReLU and variants) and why ReLU helped; vanishing/exploding gradients.
- Loss functions (cross-entropy, MSE), optimizers (SGD, momentum, RMSProp, **Adam**), learning-rate schedules.
- Regularization in DL: dropout, batch/layer norm, early stopping, weight decay, data augmentation.
- Weight initialization; the role of mini-batches; epochs vs. iterations.
- PyTorch mechanics: tensors, autograd, `nn.Module`, the training loop.

**What interviewers probe:** explain backprop and the vanishing-gradient problem; why batch norm / dropout help; how Adam differs from plain SGD.

**Sample interview questions**
1. **Explain backpropagation.** — *Signal:* reverse-mode chain rule computing the loss gradient w.r.t. each weight layer by layer; the optimizer then updates weights.
2. **What causes vanishing/exploding gradients, and how do you fix them?** — *Signal:* deep nets + saturating activations; fixes include ReLU, good init (He/Xavier), batch/layer norm, residual connections, gradient clipping.
3. **How does Adam differ from plain SGD?** — *Signal:* adaptive per-parameter learning rates via first/second moment estimates with bias correction → faster convergence, less tuning.
4. **How do dropout and batch norm help training?** — *Signal:* dropout regularizes by randomly dropping units (prevents co-adaptation); batch norm normalizes activations for faster, more stable training (mild regularization).

**Core resources**
- [Andrej Karpathy — *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) — build backprop, MLPs, and a GPT from scratch; outstanding for true understanding.
- [*Dive into Deep Learning* (d2l.ai)](https://d2l.ai/) — free, interactive textbook with PyTorch code.
- [PyTorch — official tutorials ("Learn the Basics")](https://pytorch.org/tutorials/beginner/basics/intro.html).
- [3Blue1Brown — Neural Networks series](https://www.3blue1brown.com/topics/neural-networks) — visual intuition for what training does.

---

## 15. Computer Vision & CNNs
*Course material: Classes 45 (CNNs), 46 (CNN architectures & transfer learning)*

**Master these concepts**
- Convolution, kernels/filters, stride, padding, pooling, receptive field; why convolutions beat dense layers for images (parameter sharing, locality, translation equivariance).
- Classic architectures (LeNet → AlexNet → VGG → ResNet) and the ideas they introduced (depth, skip/residual connections).
- **Transfer learning** and fine-tuning: feature extraction vs. full fine-tune, when to use which.
- Data augmentation for vision; common tasks (classification, detection, segmentation) at a conceptual level.

**What interviewers probe:** how a conv layer works and its parameter count; why ResNet's residual connections matter; how/when to do transfer learning.

**Sample interview questions**
1. **Why use convolutions instead of fully connected layers for images?** — *Signal:* parameter sharing + locality + translation equivariance → far fewer parameters and better generalization.
2. **What problem do ResNet's residual (skip) connections solve?** — *Signal:* the degradation/vanishing-gradient problem in very deep nets; identity shortcuts ease optimization.
3. **When and how do you use transfer learning?** — *Signal:* small data or a similar domain; freeze the backbone and train a new head (feature extraction), or fine-tune deeper layers with a low learning rate.

**Core resources**
- [Stanford CS231n — *Convolutional Neural Networks for Visual Recognition* (notes)](https://cs231n.github.io/) — the canonical, deeply explained CV course notes.
- [*Dive into Deep Learning* — CNN & modern CNN chapters](https://d2l.ai/chapter_convolutional-neural-networks/index.html).
- [PyTorch — Transfer Learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

---

# Part IV — NLP, Transformers & Generative AI

## 16. NLP Foundations
*Course material: Classes 47 (basics of NLP), 48 (text vectorization & embeddings)*

**Master these concepts**
- Text preprocessing: tokenization, stemming/lemmatization, stop words, n-grams.
- Classical representations: bag-of-words, **TF-IDF**, their strengths/limits.
- **Word embeddings:** Word2Vec (CBOW/skip-gram), GloVe; why dense embeddings capture semantics; cosine similarity; the analogy idea ("king − man + woman").
- Contextual vs. static embeddings (the bridge to transformers).

**What interviewers probe:** TF-IDF vs. embeddings, how Word2Vec is trained, and why context matters (motivating BERT-style embeddings).

**Sample interview questions**
1. **TF-IDF vs. word embeddings — trade-offs.** — *Signal:* TF-IDF is sparse, interpretable, no semantics; embeddings are dense, capture similarity/analogy, but need training and are less interpretable.
2. **How is Word2Vec (skip-gram) trained?** — *Signal:* predict context words from a center word; the learned weight matrix is the embeddings; negative sampling for efficiency.
3. **Why move from static to contextual embeddings?** — *Signal:* static gives one vector per word (polysemy problem — "bank"); contextual embeddings (BERT) vary with the sentence.

**Core resources**
- [Stanford CS224n — *NLP with Deep Learning*](https://web.stanford.edu/class/cs224n/) — lectures, notes, and assignments; the standard NLP course.
- [Jay Alammar — *The Illustrated Word2Vec*](https://jalammar.github.io/illustrated-word2vec/) — the clearest visual explanation of embeddings.
- [Hugging Face — LLM/NLP Course, early chapters](https://huggingface.co/learn/llm-course) — tokenization and the modern NLP pipeline.

---

## 17. Sequence Models (RNN, LSTM, Seq2Seq, Attention)
*Course material: Classes 50 (RNN/LSTM), 51 (Seq2Seq & attention)*

**Master these concepts**
- RNNs: sequential processing, hidden state, backprop-through-time, the vanishing-gradient problem.
- **LSTM/GRU:** gates, the cell state, how they mitigate long-range dependency issues.
- **Seq2Seq** (encoder–decoder) and the information-bottleneck problem.
- **Attention mechanism:** the intuition (learned, weighted lookups), how it fixes the bottleneck — the conceptual bridge to transformers.

**What interviewers probe:** why RNNs struggle with long sequences, how LSTM gates help, and what attention computes and why it was a breakthrough.

**Sample interview questions**
1. **Why do vanilla RNNs struggle with long sequences?** — *Signal:* vanishing/exploding gradients through time prevent learning long-range dependencies.
2. **How do LSTM gates help?** — *Signal:* a cell state plus forget/input/output gates regulate information flow and preserve gradients over long spans.
3. **What problem does attention solve in seq2seq?** — *Signal:* the fixed-length context-vector bottleneck; attention lets the decoder weight all encoder states at each output step.

**Core resources**
- [Christopher Olah — *Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — the definitive visual explainer.
- [Jay Alammar — *Visualizing A Neural Machine Translation Model (Seq2Seq with Attention)*](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/).
- [Stanford CS224n — RNN/LSTM/attention lectures](https://web.stanford.edu/class/cs224n/).

---

## 18. Transformers & LLMs
*Course material: Classes 52 (Transformer, BERT, GPT, T5, LLM), 53 (Hugging Face tutorial)*

**Master these concepts**
- The **transformer** architecture: self-attention, multi-head attention, query/key/value, positional encoding, feed-forward blocks, residual connections + layer norm; why it parallelizes where RNNs can't.
- Encoder-only (**BERT**), decoder-only (**GPT**), encoder–decoder (**T5/BART**) — and which tasks each suits.
- Pretraining objectives (masked LM vs. causal LM), tokenization (BPE/WordPiece), context windows.
- LLM concepts: parameters/scale, emergent abilities, **fine-tuning vs. prompting**, instruction tuning & RLHF (conceptual), parameter-efficient fine-tuning (LoRA/PEFT), quantization, hallucination.

**What interviewers probe:** explain self-attention end to end (including Q/K/V and why scaling/softmax); BERT vs. GPT; when to fine-tune vs. prompt vs. do RAG.

**Sample interview questions**
1. **Explain self-attention using Q, K, V.** — *Signal:* each token's query is compared to all keys via scaled dot-product → softmax weights → weighted sum of values; captures context and is parallelizable.
2. **Why scale by √dₖ, and why use multi-head attention?** — *Signal:* scaling keeps dot products from saturating the softmax; multiple heads learn different relation subspaces.
3. **BERT vs. GPT — architecture and typical use.** — *Signal:* BERT is encoder-only/bidirectional (masked LM) → understanding/classification; GPT is decoder-only/causal → generation.
4. **When do you fine-tune vs. prompt vs. use RAG?** — *Signal:* prompt/few-shot for quick general tasks; RAG for fresh/proprietary factual grounding; fine-tune for style/format/domain behavior or latency when you have data.

**Core resources**
- [Jay Alammar — *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) and [*The Illustrated GPT-2*](https://jalammar.github.io/illustrated-gpt2/) — essential, interview-favorite explainers.
- [*Attention Is All You Need* (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) and [*BERT* (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) — read the originals at least once.
- [Hugging Face — LLM Course](https://huggingface.co/learn/llm-course) — transformers, fine-tuning, and the `transformers` library, hands-on.
- [Karpathy — *Let's build GPT* (within Zero to Hero)](https://karpathy.ai/zero-to-hero.html) — implement a transformer from scratch.

---

## 19. Generative AI: Prompting, RAG & Agents
*Course material: Classes 54–55 (prompt engineering), 56 (RAG), 57 (AI agents & agentic AI)*

**Master these concepts**
- **Prompt engineering:** zero/few-shot, chain-of-thought, role/system prompts, structured output, decoding params (temperature, top-p), self-consistency; common failure modes and how to mitigate them.
- **RAG:** why it exists (grounding, fresh/proprietary knowledge, reduced hallucination); the pipeline — chunking → **embeddings** → **vector store** → retrieval (dense/sparse/hybrid) → reranking → generation; evaluation (retrieval quality vs. answer quality); chunking strategies and their trade-offs.
- **Vector search:** embeddings, similarity metrics, approximate nearest neighbor (concept), common stores (FAISS, Pinecone, Chroma, pgvector).
- **AI agents / agentic AI:** the augmented-LLM loop (reason → act/tool-use → observe), tool/function calling, planning, memory, multi-agent vs. single-agent, the ReAct pattern; **when *not* to build an agent**; orchestration frameworks (LangGraph/LangChain, LlamaIndex).
- **Evaluation & safety for LLM apps:** offline/online eval, LLM-as-judge, guardrails, prompt injection, cost/latency/quality trade-offs.

**What interviewers probe:** design a RAG system for a given use case and justify each component; when to use prompting vs. RAG vs. fine-tuning; what makes an agent reliable; how you'd evaluate an LLM application.

**Sample interview questions**
1. **Walk through a RAG pipeline and where it can fail.** — *Signal:* chunk → embed → store → retrieve (→ rerank) → generate; failure points: poor chunking, weak embeddings/retrieval, irrelevant context, missing grounding/citations, no evaluation.
2. **How do you reduce hallucinations in an LLM application?** — *Signal:* RAG grounding with citations, constrained prompts, lower temperature, output validation/guardrails, allow the model to abstain, and an eval harness.
3. **When should you NOT build an agent?** — *Signal:* for well-defined, deterministic tasks a single call or fixed workflow is cheaper and more reliable; agents add latency, cost, and unpredictability.
4. **How would you evaluate a RAG/LLM system?** — *Signal:* separate retrieval metrics (recall@k, MRR) from generation metrics (faithfulness, relevance); use LLM-as-judge + human eval, offline and online, and track cost/latency.

**Core resources**
- [Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/) — comprehensive, model-agnostic; covers prompting, RAG, and agents.
- [Anthropic — *Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents) and its [reference implementations (cookbook)](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents) — the most-cited practical guide to agent patterns.
- [RAG — original paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) and a [comprehensive RAG survey (Gao et al., 2023)](https://arxiv.org/abs/2312.10997).
- [Hugging Face — Agents Course](https://huggingface.co/learn/agents-course) and [LLM Course (RAG & fine-tuning chapters)](https://huggingface.co/learn/llm-course).
- [LangChain](https://python.langchain.com/docs/introduction/) and [LlamaIndex](https://docs.llamaindex.ai/) docs — the orchestration frameworks you'll be expected to know by name.
- Prompting docs worth skimming: [Anthropic prompt engineering](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview) · [OpenAI prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering).

---

# Part V — General Skills

This skills are not tied to a particular topic. These are foundational, topic-independent skills needed across the whole interview process.

## A. ML & GenAI System Design
*(Not a single class — synthesizes the whole course; increasingly central for both DS/ML and AI Engineer interviews.)*

**Master these concepts**
- A repeatable framework: **clarify requirements → define the ML objective & metrics → data & features → model choice → training & evaluation → serving → monitoring & iteration.**
- Translating a business problem into an ML problem; offline vs. online metrics; guardrail metrics.
- Scale, latency, and cost trade-offs; feedback loops and retraining; failure modes.
- For **GenAI design**: prompt vs. RAG vs. fine-tune decision; retrieval architecture; evaluation harness; guardrails; cost/latency budgeting; caching.

**What interviewers probe:** structured thinking out loud, sensible trade-offs, and the questions you ask before designing.

**Sample interview questions**
1. **Design a recommendation system for an e-commerce homepage.** — *Signal:* clarify the goal/metric (CTR vs. revenue), use candidate generation + ranking, define features, handle cold start, plan offline + online eval, respect latency, and close the feedback loop.
2. **Design a customer-support assistant over a company knowledge base.** — *Signal:* RAG architecture (ingestion/chunking, embeddings, retrieval + rerank, grounded generation with citations), guardrails, evaluation, escalation to humans, cost/latency.
3. **How do you decide between a simpler and a more complex model in a design?** — *Signal:* start from a baseline; add complexity only for measurable gains; weigh latency, cost, interpretability, and maintenance.

**Core resources**
- [Chip Huyen — *Machine Learning Systems Design* (free notes + question set)](https://huyenchip.com/machine-learning-systems-design/toc.html).
- [*Designing Machine Learning Systems* (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — the book most cited for these rounds.
- [Evidently AI — ML system design interview questions](https://www.evidentlyai.com/ml-system-design) — worked examples to practice against.
- [*Designing Data-Intensive Applications* (Kleppmann)](https://dataintensive.net/) — for the data/systems plumbing behind ML platforms.

---

## B. SQL, Coding & DSA
*(SQL isn't a dedicated class but is expected in most DS interviews; Python coding builds on Classes 1–9.)*

**Master these concepts**
- **SQL:** joins, `GROUP BY`/aggregations, **window functions** (running totals, rankings), CTEs, subqueries, date handling — the bread and butter of analytics screens.
- **Python coding:** clean implementation under time limits; common patterns (hashing, two pointers, sorting).
- **DSA (role-dependent):** arrays/strings, hash maps, recursion, trees/graphs basics, complexity analysis — heavier for ML-engineer roles.

**What interviewers probe:** writing a correct window-function query; implementing a small algorithm cleanly; reasoning about complexity.

**Sample interview questions**
1. **Find the second-highest salary in each department.** — *Signal:* window function — `DENSE_RANK()`/`ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)`, then filter to rank = 2.
2. **`WHERE` vs. `HAVING`, and `INNER` vs. `LEFT JOIN`.** — *Signal:* `WHERE` filters before aggregation, `HAVING` after; `LEFT JOIN` keeps unmatched left rows as NULLs.
3. **Compute a 7-day rolling average of daily sales.** — *Signal:* `AVG(sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`.

**Core resources**
- [DataLemur — SQL & DS interview questions](https://datalemur.com/) — DS-flavored SQL practice (founded by an ex-FAANG DS).
- [StrataScratch](https://www.stratascratch.com/) — real DS/analytics SQL & Python questions from company interviews.
- [LeetCode](https://leetcode.com/) — coding/DSA; use the SQL track too.
- [NeetCode](https://neetcode.io/) — structured DSA roadmap with clear explanations.
- [Mode — SQL Tutorial](https://mode.com/sql-tutorial/) — quick, applied SQL refresher.

---

## C. Behavioral, Projects & Communication

**Master these concepts**
- A crisp **project narrative** for 2–3 projects (use the capstones): problem → data → approach → key decisions/trade-offs → results/impact → lessons.
- The **STAR** format (Situation, Task, Action, Result) for behavioral questions.
- Communicating technical depth to non-technical stakeholders; explaining trade-offs and uncertainty honestly.
- Ownership, collaboration, and how you handled a failure or a disagreement.

**What interviewers probe:** depth and honesty in the project deep-dive, and whether you can explain *why* you made each modeling choice.

**Sample interview questions**
1. **Walk me through a project you're proud of.** — *Signal:* STAR structure — problem framing, the *specific* decisions and trade-offs you made, measurable impact, and lessons learned.
2. **Tell me about a time your model failed (or you were wrong).** — *Signal:* take ownership, identify the root cause, describe the fix and the systemic change you made; honesty over blame.
3. **Explain a technical result to a non-technical stakeholder.** — *Signal:* lead with the decision/impact, avoid jargon, convey uncertainty, and tie it to the business question.

**Core resources**
- Use our [Capstone projects](https://github.com/skarma91/logicmojo-data-science-ai-nov-2025/tree/main/Capstone_projects) as the backbone of your project stories.
- [Chip Huyen — *Introduction to Machine Learning Interviews* book](https://huyenchip.com/ml-interviews-book/) — includes excellent guidance on the non-technical sides of the process.

---

## D. General Resource Hub & Practice Platforms

**Foundational books (free or standard)**
- [*An Introduction to Statistical Learning* (ISL)](https://www.statlearning.com/) — free; the single best classical-ML reference for interviews.
- [*The Elements of Statistical Learning* (ESL)](https://hastie.su.domains/ElemStatLearn/) — free; deeper/more mathematical companion to ISL.
- [*Mathematics for Machine Learning*](https://mml-book.github.io/) — free.
- [*Dive into Deep Learning* (d2l.ai)](https://d2l.ai/) — free, code-first DL textbook.
- [*Hands-On Machine Learning* (Géron) — code](https://github.com/ageron/handson-ml3) — the standard practical ML book's notebooks.

**End-to-end courses**
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Hugging Face — LLM Course](https://huggingface.co/learn/llm-course) and [Agents Course](https://huggingface.co/learn/agents-course)
- [Stanford CS229 (ML)](https://cs229.stanford.edu/), [CS231n (CV)](https://cs231n.github.io/), [CS224n (NLP)](https://web.stanford.edu/class/cs224n/)
- [Made With ML — MLOps](https://madewithml.com/)

**Explainers worth bookmarking**
- [StatQuest (Josh Starmer)](https://www.youtube.com/@statquest) — best-in-class for classical ML/stats intuition.
- [3Blue1Brown](https://www.3blue1brown.com/) — math & neural-network intuition.
- [Jay Alammar](https://jalammar.github.io/) — illustrated NLP/transformers.
- [Distill.pub](https://distill.pub/) — visual deep-dives on select DL topics.

**Interview-specific question banks & practice**
- [Chip Huyen — *Introduction to Machine Learning Interviews* (free book)](https://huyenchip.com/ml-interviews-book/)
- [*Ace the Data Science Interview* (book)](https://www.acethedatascienceinterview.com/) — stats, ML, SQL, and case questions.
- [DataLemur](https://datalemur.com/), [StrataScratch](https://www.stratascratch.com/), [LeetCode](https://leetcode.com/), [NeetCode](https://neetcode.io/)
- [Kaggle](https://www.kaggle.com/) — competitions + [Kaggle Learn](https://www.kaggle.com/learn) micro-courses; great for a portfolio.

---

## Suggested 6-week revision plan (adapt to your timeline)

| Week | Focus |
|------|-------|
| 1 | Python/Pandas/SQL fluency + Math (linear algebra, calculus, optimization) |
| 2 | Probability + Statistics & inference (incl. A/B testing) |
| 3 | Classical ML: regression, classification, bias–variance/regularization, evaluation |
| 4 | Ensembles/boosting, dimensionality reduction, unsupervised, feature engineering, deployment |
| 5 | Deep learning, CNNs, NLP foundations, time series, sequence models |
| 6 | Transformers & LLMs, GenAI (prompting/RAG/agents), system design, behavioral + mock interviews |

> Run **coding + SQL practice daily** in parallel, and do **at least 2–3 mock interviews** (one ML system design, one GenAI design) in the final stretch.

---

*Tip: the sample questions here are a starting set, not a question bank — interviewers will go deeper, so practice the follow-ups ("why?", "what if the data were imbalanced?", "how would you scale it?"). Your fastest reference is always your own notebooks and notes in the [course repo](https://github.com/skarma91/logicmojo-data-science-ai-nov-2025); the external links are for going deeper, not a replacement for re-deriving the material yourself.*
