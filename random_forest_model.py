
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading LendingClub dataset...")

df = pd.read_csv("loan_modeling_base.csv", low_memory=False)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# --------------------------------------------------

target = "default_flag"

numeric_features = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
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

numeric_features = [
    c for c in numeric_features if c in df.columns
]

categorical_features = [
    c for c in categorical_features if c in df.columns
]

features = numeric_features + categorical_features

X = df[features].copy()
y = df[target].astype(int)

# Convert numeric columns
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Convert categorical columns safely
for col in categorical_features:
    X[col] = X[col].astype("string").to_numpy(
        dtype=object,
        na_value=np.nan
    )

print("\nFeatures selected:", len(features))
print("Target distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 4. PREPROCESSING
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
# 5. RANDOM FOREST MODEL
# --------------------------------------------------

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest model...")

rf_model.fit(X_train, y_train)

print("Random Forest training completed!")

# --------------------------------------------------
# 7. PREDICTIONS
# --------------------------------------------------

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 8. EVALUATION
# --------------------------------------------------

print("\n========== RANDOM FOREST RESULTS ==========")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fully Paid", "Default"]
))

roc = roc_auc_score(y_test, y_prob)
pr = average_precision_score(y_test, y_prob)

print("ROC-AUC:", round(roc, 4))
print("PR-AUC:", round(pr, 4))

# --------------------------------------------------
# 9. THRESHOLD COMPARISON
# --------------------------------------------------

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

    predictions = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

# --------------------------------------------------
# 10. SAVE MODEL
# --------------------------------------------------

joblib.dump(rf_model, "lendingclub_random_forest.pkl")

print("\nModel saved as lendingclub_random_forest.pkl")
print("\nRandom Forest project completed!")
