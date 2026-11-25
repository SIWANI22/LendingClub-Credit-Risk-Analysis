
import pandas as pd

FILE_PATH = r"C:\Users\siwan\OneDrive\Documents\Machine Learnig (IN)\loan.csv"

print("Scanning the complete Lending Club dataset...")

status_counts = {}

for chunk in pd.read_csv(FILE_PATH, chunksize=100000):

    counts = chunk["loan_status"].value_counts()

    for status, count in counts.items():
        status_counts[status] = status_counts.get(status, 0) + count

results = pd.Series(status_counts).sort_values(ascending=False)

print("\n========== COMPLETE LOAN STATUS DISTRIBUTION ==========")
print(results)

print("\n========== TOTAL RECORDS ==========")
print(results.sum())

print("\n========== DEFAULT RECORDS ==========")
print("Charged Off:", status_counts.get("Charged Off", 0))

print("\n========== ANALYSIS COMPLETE ==========")
