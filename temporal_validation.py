
import pandas as pd
import numpy as np

from pathlib import Path

# 1. Load the prepared LendingClub dataset
DATA_FILE = Path("loan_modeling_base.csv")

df = pd.read_csv(DATA_FILE, low_memory=False)

print("DATASET LOADED")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Convert loan issue month into a date
df["issue_date"] = pd.to_datetime(
    df["issue_d"],
    format="%b-%y",
    errors="coerce"
)

# 3. Inspect the date coverage
print("\nLOAN ISSUE DATE SUMMARY")
print("Earliest:", df["issue_date"].min())
print("Latest:", df["issue_date"].max())
print("Missing issue dates:", df["issue_date"].isna().sum())

print("\nLOAN COUNTS BY YEAR")
print(df["issue_date"].dt.year.value_counts().sort_index())

# 4. Check the target distribution
print("\nDEFAULT TARGET DISTRIBUTION")
print(df["default_flag"].value_counts())

print("\nDEFAULT RATE (%)")
print(round(df["default_flag"].mean() * 100, 2))

# 5. Create chronological training and testing datasets

train_df = df[
    df["issue_date"] <= "2016-12-31"
].copy()

test_df = df[
    df["issue_date"] >= "2018-01-01"
].copy()

# 6. Display dataset sizes

print("\nTEMPORAL TRAIN / TEST SPLIT")

print("Training period: Oct 2015 - Dec 2016")
print("Training records:", len(train_df))

print("\nTesting period: Jan 2018 - Dec 2018")
print("Testing records:", len(test_df))

# 7. Compare default rates

print("\nDEFAULT RATE BY PERIOD")

print(
    "Training default rate:",
    round(train_df["default_flag"].mean() * 100, 2), "%"
)

print(
    "Testing default rate:",
    round(test_df["default_flag"].mean() * 100, 2), "%"
)

# 8. Verify that the periods do not overlap

print("\nDATE VALIDATION")

print("Latest training date:", train_df["issue_date"].max())
print("Earliest testing date:", test_df["issue_date"].min())

# 9. Compare class distribution across periods

print("\nCLASS DISTRIBUTION: TRAINING DATA")
print(train_df["default_flag"].value_counts().sort_index())

print("\nCLASS DISTRIBUTION: TESTING DATA")
print(test_df["default_flag"].value_counts().sort_index())

print("\nCLASS DISTRIBUTION AS PERCENTAGES")

print(
    "Training:"
)
print(
    (train_df["default_flag"].value_counts(normalize=True) * 100).round(2)
)

print(
    "\nTesting:"
)
print(
    (test_df["default_flag"].value_counts(normalize=True) * 100).round(2)
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)



# 10. Define model features

numeric_features = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "revol_util",
    "open_acc",
    "int_rate",
    "installment"
]
numeric_features_reduced = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "revol_util",
    "open_acc",
    "int_rate"
]  
categorical_features_reduced = [
    "grade",
    "term"
]
categorical_features_subgrade = [
    "sub_grade",
    "term"
]
categorical_features = [
    "grade",
    "sub_grade",
    "term"
]
numeric_features_final = numeric_features_reduced

categorical_features_final = categorical_features_subgrade
target = "default_flag"

print("\nMODEL FEATURE SELECTION")
print("Numeric features:", numeric_features_final)
print("Categorical features:", categorical_features_final)
print("Target variable:", target)



# 11. Prepare numerical and categorical data

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features_final),
        ("cat", categorical_transformer, categorical_features_final)
    ]
)

print("\nPREPROCESSING PIPELINE CREATED")


# 12. Separate features (X) and target (y)

X_train = train_df[numeric_features_final + categorical_features_final]
y_train = train_df[target]

X_test = test_df[numeric_features_final + categorical_features_final]
y_test = test_df[target]


# 13. Create the Logistic Regression model

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])


# 14. Train the model using historical data

print("\nTRAINING LOGISTIC REGRESSION MODEL...")

logistic_model.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# 15. Generate predictions on unseen testing data

y_pred = logistic_model.predict(X_test)

# Probability of default (class 1)
y_proba = logistic_model.predict_proba(X_test)[:, 1]


# 16. Evaluate model performance

print("\n" + "=" * 50)
print("TEMPORAL LOGISTIC REGRESSION RESULTS")
print("=" * 50)

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\nROC-AUC SCORE")
print(round(roc_auc_score(y_test, y_proba), 4))

print("\nPR-AUC SCORE (Average Precision)")
print(round(average_precision_score(y_test, y_proba), 4))


# 17. Evaluate different classification thresholds

from sklearn.metrics import precision_score, recall_score, f1_score

print("\nTHRESHOLD COMPARISON")
print("-" * 65)

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

    threshold_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, threshold_pred)
    recall = recall_score(y_test, threshold_pred)
    f1 = f1_score(y_test, threshold_pred)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )
print("\nBUSINESS IMPACT BY THRESHOLD")
print("-" * 75)

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

    threshold_pred = (y_proba >= threshold).astype(int)

    total_flagged = threshold_pred.sum()
    actual_defaults_caught = (
        (threshold_pred == 1) & (y_test == 1)
    ).sum()

    false_alarms = (
        (threshold_pred == 1) & (y_test == 0)
    ).sum()

    missed_defaults = (
        (threshold_pred == 0) & (y_test == 1)
    ).sum()

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Loans flagged as risky: {total_flagged:,}")
    print(f"Actual defaults caught: {actual_defaults_caught:,}")
    print(f"False alarms: {false_alarms:,}")
    print(f"Defaults missed: {missed_defaults:,}")
    print("\nBUSINESS IMPACT BY THRESHOLD")
print("-" * 75)

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

    threshold_pred = (y_proba >= threshold).astype(int)

    total_flagged = threshold_pred.sum()

    actual_defaults_caught = (
        (threshold_pred == 1) & (y_test == 1)
    ).sum()

    false_alarms = (
        (threshold_pred == 1) & (y_test == 0)
    ).sum()

    missed_defaults = (
        (threshold_pred == 0) & (y_test == 1)
    ).sum()

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Loans flagged as risky: {total_flagged:,}")
    print(f"Actual defaults caught: {actual_defaults_caught:,}")
    print(f"False alarms: {false_alarms:,}")
    print(f"Defaults missed: {missed_defaults:,}")

print("\nNUMERIC FEATURE CORRELATION ANALYSIS")
print("-" * 65)

# Use training data only for this diagnostic
correlation_matrix = X_train[numeric_features_final].corr().abs()

# Keep only the upper triangle to avoid duplicate pairs
upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

# Display feature pairs with correlation >= 0.70
high_correlations = (
    upper_triangle.stack()
    .sort_values(ascending=False)
)

print("\nHighly correlated feature pairs (>= 0.70):")

print(
    high_correlations[high_correlations >= 0.70]
)

# ============================================================
# MODEL B: REMOVE INSTALLMENT FEATURE
# ============================================================

print("\nMODEL B: WITHOUT INSTALLMENT")
print("-" * 65)

# Prepare Model B data
X_train_reduced = train_df[numeric_features_reduced + categorical_features]
X_test_reduced = test_df[numeric_features_reduced + categorical_features]

# Create a separate preprocessing pipeline
numeric_transformer_reduced = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer_reduced = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_reduced = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_reduced, numeric_features_reduced),
        ("cat", categorical_transformer_reduced, categorical_features)
    ]
)

# Build Model B
logistic_model_reduced = Pipeline(steps=[
    ("preprocessor", preprocessor_reduced),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train Model B
logistic_model_reduced.fit(X_train_reduced, y_train)

# Evaluate Model B
y_proba_reduced = logistic_model_reduced.predict_proba(X_test_reduced)[:, 1]
y_pred_reduced = logistic_model_reduced.predict(X_test_reduced)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_reduced))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_reduced))

print("ROC-AUC:", roc_auc_score(y_test, y_proba_reduced))
print("PR-AUC:", average_precision_score(y_test, y_proba_reduced))


# ============================================================
# CATEGORICAL FEATURE RELATIONSHIP ANALYSIS
# ============================================================

print("\nCATEGORICAL FEATURE RELATIONSHIP ANALYSIS")
print("-" * 65)

# 1. Check how sub_grade relates to grade
print("\nSUB-GRADE DISTRIBUTION WITHIN EACH GRADE")

grade_subgrade = pd.crosstab(
    train_df["grade"],
    train_df["sub_grade"],
    normalize="index"
)

print(grade_subgrade.round(2))

# 2. Analyze interest rates across grades
print("\nINTEREST RATE BY GRADE")

grade_interest = train_df.groupby("grade")["int_rate"].agg(
    ["count", "min", "median", "max"]
)

print(grade_interest.round(2))

# 3. Analyze interest rates across sub-grades
print("\nINTEREST RATE BY SUB-GRADE")

subgrade_interest = train_df.groupby("sub_grade")["int_rate"].agg(
    ["count", "min", "median", "max"]
)

print(subgrade_interest.round(2))


# ============================================================
# MODEL C: WITHOUT SUB_GRADE
# ============================================================

print("\nMODEL C: WITHOUT SUB_GRADE")
print("-" * 65)

# Prepare Model C data
X_train_c = train_df[numeric_features_reduced + categorical_features_reduced]
X_test_c = test_df[numeric_features_reduced + categorical_features_reduced]

# Create separate preprocessing pipelines
numeric_transformer_c = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer_c = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_c = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_c, numeric_features_reduced),
        ("cat", categorical_transformer_c, categorical_features_reduced)
    ]
)

# Build Model C
logistic_model_c = Pipeline(steps=[
    ("preprocessor", preprocessor_c),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train Model C
logistic_model_c.fit(X_train_c, y_train)

# Evaluate Model C
y_proba_c = logistic_model_c.predict_proba(X_test_c)[:, 1]
y_pred_c = logistic_model_c.predict(X_test_c)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_c))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_c))

print("ROC-AUC:", roc_auc_score(y_test, y_proba_c))
print("PR-AUC:", average_precision_score(y_test, y_proba_c))


# ============================================================
# MODEL D: WITHOUT GRADE
# ============================================================

print("\nMODEL D: WITHOUT GRADE")
print("-" * 65)

# Prepare Model D data
X_train_d = train_df[numeric_features_reduced + categorical_features_subgrade]
X_test_d = test_df[numeric_features_reduced + categorical_features_subgrade]

# Create separate preprocessing pipelines
numeric_transformer_d = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer_d = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_d = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_d, numeric_features_reduced),
        ("cat", categorical_transformer_d, categorical_features_subgrade)
    ]
)

# Build Model D
logistic_model_d = Pipeline(steps=[
    ("preprocessor", preprocessor_d),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train Model D
logistic_model_d.fit(X_train_d, y_train)

# Evaluate Model D
y_proba_d = logistic_model_d.predict_proba(X_test_d)[:, 1]
y_pred_d = logistic_model_d.predict(X_test_d)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_d))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_d))

print("ROC-AUC:", roc_auc_score(y_test, y_proba_d))
print("PR-AUC:", average_precision_score(y_test, y_proba_d))
