import pandas as pd

FILE_PATH = r"C:\Users\siwan\OneDrive\Documents\Machine Learnig (IN)\loan.csv"

print("Starting...")
df = pd.read_csv(FILE_PATH, nrows=50000)
print("Loaded!")
print("Shape (rows, columns):", df.shape)

FILE_PATH = r"C:\Users\siwan\OneDrive\Documents\Machine Learnig (IN)\loan.csv"

df = pd.read_csv(FILE_PATH, nrows=50000)
# 2. Dataset overview
print("\n========== DATASET OVERVIEW ==========")
print("Dataset Shape:", df.shape)

# 3. First 5 records
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# 4. Column names
print("\n========== COLUMN NAMES ==========")
for col in df.columns:
    print(col)

# 5. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)    

# 6. Missing values
print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# 7. Duplicate records
print("\n========== DUPLICATES ==========")
print("Duplicate Rows:", df.duplicated().sum())

# 8. Loan status distribution
print("\n========== LOAN STATUS ==========")
if "loan_status" in df.columns:
    print(df["loan_status"].value_counts(dropna=False))
else:
    print("loan_status column not found. Check column names.")

print("\n========== ANALYSIS COMPLETE ==========")

