import pandas as pd

# Define file paths (adjust these paths as needed)
abtiters_path = "../data/Measles/antibody_df.csv"
cyto_data_path = "../data/Measles/cyto_data.csv"
output_path = "../data/Measles/cyto_data_sorted.csv"

# Load the antibody titers data
abtiters_df = pd.read_csv(abtiters_path)

# Extract the reference order based on the "Vaccinee" column
reference_order = abtiters_df["Vaccinee"].tolist()

# Load the cytometry data
cyto_df = pd.read_csv(cyto_data_path)

# Check if "Vaccinee" is a column in cyto_df. If not, you might have to reset the index.
if "Vaccinee" not in cyto_df.columns:
    cyto_df.reset_index(inplace=True)
    # If the Vaccinee IDs are in the index, rename the index column.
    cyto_df.rename(columns={"index": "Vaccinee"}, inplace=True)

# Set "Vaccinee" as the index and reindex according to reference_order
cyto_df_sorted = cyto_df.set_index("Vaccinee").reindex(reference_order).reset_index()

# Save the sorted DataFrame to a new CSV file
cyto_df_sorted.to_csv(output_path, index=False)

print(f"Sorted cytometry data saved to {output_path}")