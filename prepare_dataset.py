
import pandas as pd
import os

# Original dataset
FILE_PATH = r"C:\Users\siwan\OneDrive\Documents\Machine Learnig (IN)\loan.csv"

# New output dataset
OUTPUT_PATH = r"C:\Users\siwan\OneDrive\Documents\Machine Learnig (IN)\loan_modeling_base.csv"

print("Starting dataset preparation...")

total_rows = 0
selected_rows = 0

# Process data in chunks to manage memory
for chunk in pd.read_csv(FILE_PATH, chunksize=100000):

    total_rows += len(chunk)

    # Keep only loans with known final outcomes
    chunk = chunk[
        chunk["loan_status"].isin(
            ["Fully Paid", "Charged Off", "Default"]
        )
    ].copy()

    # Create binary target variable
    chunk["default_flag"] = chunk["loan_status"].map({
        "Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1
    })

    selected_rows += len(chunk)

    # Save processed chunk
    chunk.to_csv(
        OUTPUT_PATH,
        mode="w" if selected_rows == len(chunk) else "a",
        header=(selected_rows == len(chunk)),
        index=False
    )

    print(f"Processed {total_rows:,} records...")

print("\n========== DATASET PREPARATION COMPLETE ==========")

print("Total original records:", f"{total_rows:,}")
print("Selected completed loans:", f"{selected_rows:,}")

# Verify the new dataset
df = pd.read_csv(OUTPUT_PATH)

print("\nTarget distribution:")
print(df["default_flag"].value_counts())

print("\nTarget percentages:")
print(df["default_flag"].value_counts(normalize=True) * 100)

print("\nNew dataset shape:", df.shape)

print("\nSaved to:", OUTPUT_PATH)
