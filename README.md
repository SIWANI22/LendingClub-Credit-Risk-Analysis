# LendingClub Credit Risk Analysis

## Project Overview

This project uses historical LendingClub loan data to explore credit risk and predict whether a loan will result in a default or charge-off.

The primary objective is to identify patterns associated with loan defaults and build machine learning models that can help assess credit risk. The project combines data cleaning, exploratory analysis, feature selection, preprocessing, model development, and performance evaluation.

## Business Problem

Lending institutions face the challenge of evaluating borrower risk while balancing lending opportunities and potential credit losses.

This project investigates:

- Which borrower and loan characteristics are associated with default?
- How effectively can machine learning distinguish between fully paid and defaulted loans?
- How do different classification models compare?
- What trade-offs exist between identifying more defaults and avoiding false alarms?

## Dataset and Preparation

The original dataset contains approximately 1 million loan records.

For the completed-loan modeling analysis, the following loan statuses were retained:

- Fully Paid
- Charged Off
- Default

Loans with unresolved statuses, such as Current, Late, and In Grace Period, were excluded.

**Modeling dataset:**
- 425,835 loan records
- 146 columns before final feature selection
- 331,528 fully paid loans
- 94,307 charged-off or defaulted loans

The target variable, `default_flag`, was created:

- `0` = Fully Paid
- `1` = Charged Off or Default

The resulting dataset contains approximately 77.85% non-default loans and 22.15% default loans.

## Feature Selection

The modeling process uses borrower, loan, and credit-related characteristics.

Selected numerical features include:

- Loan amount
- Interest rate
- Installment
- Annual income
- Debt-to-income ratio (DTI)
- Delinquencies in the past two years
- Recent credit inquiries
- Open accounts
- Public records
- Revolving balance
- Revolving utilization
- Total accounts

Selected categorical features include:

- Loan term
- Loan grade and sub-grade
- Employment length
- Home ownership
- Verification status
- Loan purpose
- State
- Initial listing status
- Application type

Potential data leakage was considered during feature selection. Variables reflecting subsequent payments, recoveries, collections, and loan outcomes were excluded.

**Important modeling consideration:** Interest rate, grade, and sub-grade are available after loan pricing or issuance. Therefore, the current model is better interpreted as a post-pricing or post-issuance risk analysis, rather than a pure application-time underwriting model.

## Data Preprocessing

The preprocessing workflow includes:

- Handling missing numerical and categorical values
- Numerical feature scaling
- One-hot encoding categorical variables
- Stratified 80/20 train-test split
- Class weighting to address the imbalance between default and non-default loans

The split uses `random_state=42` for reproducibility.

## Machine Learning Models

Two classification models were developed and evaluated:

1. **Logistic Regression** — a baseline classification model.
2. **Random Forest** — an ensemble learning model designed to capture more complex relationships.

Both models use class weighting to account for the minority default class.

## Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.64 | 0.63 |
| Default Precision | 0.343 | 0.338 |
| Default Recall | 0.680 | 0.685 |
| Default F1-score | 0.456 | 0.453 |
| ROC-AUC | 0.7146 | 0.7112 |
| PR-AUC | 0.4071 | 0.4043 |

### Confusion Matrix Results

**Logistic Regression**

| | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| Actual Non-Default | 41,786 | 24,520 |
| Actual Default | 6,036 | 12,825 |

**Random Forest**

| | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| Actual Non-Default | 41,041 | 25,265 |
| Actual Default | 5,939 | 12,922 |

### Initial Findings

The two models produced relatively similar results.

- Logistic Regression achieved slightly higher ROC-AUC, PR-AUC, accuracy, precision, and F1-score.
- Random Forest achieved marginally higher default recall.
- Both models identified approximately 68% of actual defaults at the 0.50 classification threshold.

These findings suggest that the more complex Random Forest model did not provide a meaningful performance improvement over the Logistic Regression baseline in the current experiment.

## Feature Importance

The initial Random Forest feature-importance analysis identified interest rate as the most influential feature, followed by loan grade and loan term.

Other influential variables included:

- Debt-to-income ratio
- Sub-grade
- Home ownership
- Annual income
- Installment
- Loan amount
- Revolving utilization

These are model-based importance measures, not evidence of causation. Correlated variables, particularly interest rate and loan grade, should be interpreted carefully.

## Project Structure

```text
LendingClub-Credit-Risk-Analysis/
│
├── data_quality.py
├── feature_audit.py
├── feature_importance.py
├── model_preparation.py
├── prepare_dataset.py
├── random_forest_model.py
├── sample_check.py
├── status_analysis.py
├── .gitignore
└── README.md
```

The original and prepared CSV datasets are excluded from the GitHub repository.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- VS Code
- Git and GitHub

## Limitations and Future Improvements

The current results are an initial modeling benchmark. Further work is needed before treating the models as operational credit-risk tools.

Planned improvements include:

1. Permutation feature importance to evaluate feature contributions more robustly.
2. Temporal train-test validation using loan issue dates.
3. Model optimization and hyperparameter tuning.
4. Classification threshold analysis based on business costs.
5. Explainability analysis and feature-level interpretation.
6. Comparing models with and without post-pricing variables such as interest rate and loan grade.
7. Evaluating the potential business impact of false positives and false negatives.

## Conclusion

This project demonstrates an end-to-end approach to historical loan data analysis and binary credit-risk classification.

The initial comparison shows that Logistic Regression performs slightly better overall, while Random Forest identifies a marginally higher proportion of defaults.

The next phase will focus on improving validation, model interpretability, and the connection between predictive performance and lending business decisions.
