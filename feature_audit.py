
import pandas as pd

FILE_PATH = r"C:\Users\siwan\OneDrive\Documents\Machine Learnig (IN)\loan_modeling_base.csv"

df = pd.read_csv(FILE_PATH, nrows=1000, low_memory=False)

# Candidate features available around loan origination
candidate_features = [
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type"
]

# Potential leakage or post-origination columns
leakage_candidates = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "hardship_loan_status",
    "settlement_amount",
    "settlement_status",
    "debt_settlement_flag_date",
    "last_credit_pull_d"
]

available_features = [
    col for col in candidate_features if col in df.columns
]

available_leakage = [
    col for col in leakage_candidates if col in df.columns
]

print("\n========== POTENTIAL PREDICTIVE FEATURES ==========")
print(available_features)

print("\n========== POTENTIAL DATA LEAKAGE FEATURES ==========")
print(available_leakage)

# Feature-level missing value report
print("\n========== FEATURE MISSING VALUE REPORT ==========")

report = pd.DataFrame({
    "Data_Type": df[available_features].dtypes.astype(str),
    "Missing_Percentage": (
        df[available_features].isnull().mean() * 100
    ).round(2)
})

print(report.sort_values("Missing_Percentage").to_string())

print("\n========== AUDIT COMPLETE ==========")
