import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# Define file paths
hep_path = "../data/Hepatitis B/module_scores_model_input_all_self_made.csv"
measles_path = "../data/Measles/module_scores_model_input_all_self_made.csv"
combined_path = "../data/Combined/module_scores_model_input_all_self_made.csv"

# Load the CSV files
df_hep = pd.read_csv(hep_path)
df_measles = pd.read_csv(measles_path)

# Select only numeric columns from each dataset
hep_numeric = df_hep.select_dtypes(include=[np.number])
measles_numeric = df_measles.select_dtypes(include=[np.number])

# Find the common numeric columns
common_cols = list(set(hep_numeric.columns).intersection(set(measles_numeric.columns)))
common_cols.sort()

# Pick 20 random columns or fewer if less than 20
selected_cols = random.sample(common_cols, min(20, len(common_cols)))
print("Selected columns:", selected_cols)

# Subset each dataset to the selected columns
hep_subset = hep_numeric[selected_cols].copy()
measles_subset = measles_numeric[selected_cols].copy()

# Add a column indicating the dataset
hep_subset["dataset"] = "Hepatitis B"
measles_subset["dataset"] = "Measles"

# Combine into a single DataFrame
combined = pd.concat([hep_subset, measles_subset], axis=0, ignore_index=True)

# Convert from wide to long format for Seaborn
melted = combined.melt(id_vars="dataset", var_name="feature", value_name="value")

# Create the figure
plt.figure(figsize=(16, 8))
sns.boxplot(x="feature", y="value", hue="dataset", data=melted)

# Improve appearance
plt.xticks(rotation=45, ha='right')
plt.title("Comparison of Selected Features Between Hepatitis B and Measles")
plt.tight_layout()
plt.show()

for col in selected_cols:
    if melted[melted["feature"] == col]["value"].isnull().any():
        print(f"Column '{col}' has NaN values.")










# Combine the two DataFrames
df_combined = pd.concat([df_hep, df_measles], ignore_index=True)

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(combined_path), exist_ok=True)

# Save the combined DataFrame to a new CSV file
df_combined.to_csv(combined_path, index=False)
print(f"Combined CSV saved to: {combined_path}")

# Check for columns with null values
null_columns = df_combined.columns[df_combined.isnull().any()]
print("\nColumns with null values:")
for col in null_columns:
    null_count = df_combined[col].isnull().sum()
    print(f"{col}: {null_count} missing values")