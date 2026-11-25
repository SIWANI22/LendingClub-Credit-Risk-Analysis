
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
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

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading LendingClub dataset...")

df = pd.read_csv(
    "loan_modeling_base.csv",
    low_memory=False
)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. DEFINE TARGET
# --------------------------------------------------

target = "default_flag"

# Remove rows without a valid target
df = df.dropna(subset=[target])

# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

numeric_features = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc"
]

categorical_features = [
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type"
]

# Keep only features that actually exist in the dataset
numeric_features = [
    col for col in numeric_features if col in df.columns
]

categorical_features = [
    col for col in categorical_features if col in df.columns
]

features = numeric_features + categorical_features

print("\nNumerical features:", numeric_features)
print("\nCategorical features:", categorical_features)

print("\nTotal selected features:", len(features))

# --------------------------------------------------
# 4. PREPARE X AND Y
# --------------------------------------------------

X = df[features].copy()
y = df[target].astype(int)

# Convert numerical columns to numeric values
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Convert categorical columns to standard Python objects
# Replace Pandas missing values with NumPy NaN
for col in categorical_features:
    X[col] = X[col].astype("string").to_numpy(
        dtype=object,
        na_value=np.nan
    )

print("\nTarget distribution:")
print(y.value_counts())

# --------------------------------------------------
# 5. SPLIT DATA
# --------------------------------------------------

print("\nSplitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 6. BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

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
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# --------------------------------------------------
# 7. BUILD LOGISTIC REGRESSION MODEL
# --------------------------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------------------------
# 9. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# THRESHOLD COMPARISON
# --------------------------------------------------

print("\n========== THRESHOLD COMPARISON ==========")

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

for threshold in thresholds:

    threshold_predictions = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test, threshold_predictions, zero_division=0
    )

    recall = recall_score(
        y_test, threshold_predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, threshold_predictions, zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

# --------------------------------------------------
# 10. EVALUATE MODEL
# --------------------------------------------------

print("\n========== MODEL EVALUATION ==========")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fully Paid", "Default"]
))

print("ROC-AUC Score:", round(
    roc_auc_score(y_test, y_prob), 4
))

print("PR-AUC Score:", round(
    average_precision_score(y_test, y_prob), 4
))

# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

joblib.dump(model, "lendingclub_logistic_model.pkl")

print("\nModel saved as lendingclub_logistic_model.pkl")

print("\nProject baseline model completed!")
