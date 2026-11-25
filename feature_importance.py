
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LOAD TRAINED RANDOM FOREST MODEL
# --------------------------------------------------

import joblib

print("Loading Random Forest model...")

model = joblib.load("lendingclub_random_forest.pkl")

# --------------------------------------------------
# 2. EXTRACT FEATURE NAMES AND IMPORTANCE
# --------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

importance_values = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# --------------------------------------------------
# 3. SORT FEATURE IMPORTANCE
# --------------------------------------------------

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 20 IMPORTANT FEATURES ==========")

print(importance_df.head(20).to_string(index=False))

# --------------------------------------------------
# 4. VISUALIZE TOP 20 FEATURES
# --------------------------------------------------

top_features = importance_df.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 20 Features Influencing Loan Default Predictions")
plt.xlabel("Feature Importance")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig("random_forest_feature_importance.png", dpi=300)

plt.show()

# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

importance_df.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

print("\nFeature importance analysis completed!")
print("Saved feature importance CSV and chart.")
