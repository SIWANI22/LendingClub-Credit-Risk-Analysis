
import pandas as pd

FILE_PATH = r"C:\Users\siwan\OneDrive\Documents\Machine Learnig (IN)\loan_modeling_base.csv"

print("Loading modeling dataset...")

df = pd.read_csv(FILE_PATH, low_memory=False)

# 1. Dataset dimensions
print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)

# 2. Verify target variable
print("\n========== TARGET DISTRIBUTION ==========")
print(df["default_flag"].value_counts())

# 3. Check column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 4. Missing values
print("\n========== MISSING VALUE ANALYSIS ==========")

missing = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().mean() * 100).round(2)
})

missing = missing[missing["Missing_Count"] > 0]
missing = missing.sort_values(
    by="Missing_Percentage",
    ascending=False
)

print(missing.to_string())

# 5. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes.value_counts())

# 6. Duplicate check
print("\n========== DUPLICATE CHECK ==========")
print("Duplicates:", df.duplicated().sum())

# 7. Check target consistency
print("\n========== TARGET VALIDATION ==========")
print(pd.crosstab(df["loan_status"], df["default_flag"]))

print("\n========== QUALITY CHECK COMPLETE ==========")
