import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

VACCINE = "Measles"

# Step 1: Read the module scores CSV
df = pd.read_csv(f"../data/{VACCINE}/module_scores_model_input_all_self_made.csv")

# Step 2: Filter columns: keep "Vaccinee" and columns that end with "EXP0"
cols_to_keep = [col for col in df.columns if col == "Vaccinee" or col.endswith("EXP0")]
df_filtered = df[cols_to_keep].copy()

string_ = "cell cycle"
count_ = df_filtered.columns.str.contains(string_, case=False).sum()
print(f"Number of column names containing '{string_}':", count_)

# Step 3: Create a mapping from original names to new names
# For columns (excluding "Vaccinee") of the form "M15.99.Protein modification_EXP0",
# we want to get "Protein modification_0" for the first occurrence, "Protein modification_1" for the second, etc.
mapping = {}
counter = {}  # Counter for each module text
names = []

new_col_names = {}
for col in df_filtered.columns:
    if col == "Vaccinee":
        new_col_names[col] = col
    else:
        # Pattern: start with M, then digits, a dot, then digits, a dot, then capture the text before _EXP0.
        match = re.match(r"^M\d+\.\d+\.(.*)_EXP0$", col)
        if match:
            text = match.group(1).strip()  # e.g., "Protein modification"
            # Increment the counter for this text
            if text not in names:
                names.append(text)
            if text in counter:
                counter[text] += 1
            else:
                counter[text] = 0
            if counter[text] == 0:
                new_name = f"{text}"
            else:
                new_name = f"{text}.{counter[text]}"
            new_col_names[col] = new_name
            mapping[col] = new_name
        else:
            # If the pattern does not match, keep the original name
            new_col_names[col] = col

print(f"Number of distinct functionalities: '{len(names)}'")

# # Remove columns whose new names contain "TBD"
cols_to_keep_final = [col for col in df_filtered.columns if "TBD" not in new_col_names[col]]
df_filtered = df_filtered[cols_to_keep_final].copy()

# Rename the columns in the filtered DataFrame
df_filtered.rename(columns=new_col_names, inplace=True)
df_filtered.to_csv(f"../data/{VACCINE}/RNA_circos.csv", index=False)

# Optionally, create a mapping DataFrame and save it
mapping_df = pd.DataFrame(list(mapping.items()), columns=["original_name", "new_name"])
mapping_df.to_csv(f"../data/{VACCINE}/RNA_circos_mapping.csv", index=False)

# Separate out the "Vaccinee" column for reattachment later
vaccinee_series = df_filtered["Vaccinee"].copy()
data_only = df_filtered.drop(columns=["Vaccinee"]).copy()


# Function to get the "base name" of a column, ignoring any suffixes like ".1", ".2", etc.
# e.g., "Protein synthesis.1" -> "Protein synthesis"
def get_base_name(col):
    match = re.match(r"^(.*?)(?:\.\d+)?$", col)
    if match:
        return match.group(1)
    else:
        return col


# 2) Group columns by their base name
grouped_columns = {}
for col in data_only.columns:
    base = get_base_name(col)
    if base not in grouped_columns:
        grouped_columns[base] = []
    grouped_columns[base].append(col)

# 3) For each group of columns with the same base name, do PCA => 1 component
#    We'll also standardize each group before PCA, so each column has mean=0, var=1
compressed_data = pd.DataFrame(index=data_only.index)  # same rows (subjects) as original

for base_name, cols in grouped_columns.items():
    # Subset the data
    subset = data_only[cols].copy()

    # Standardize these columns
    scaler = StandardScaler()
    subset_scaled = scaler.fit_transform(subset)

    # PCA to 1 component
    pca = PCA(n_components=1)
    pc_scores = pca.fit_transform(subset_scaled)  # shape: (n_samples, 1)

    # Flatten and store in compressed_data under the base_name
    compressed_data[base_name] = pc_scores[:, 0]

# 4) Reattach the "Vaccinee" column at the front
compressed_data.insert(0, "Vaccinee", vaccinee_series)

# 5) Save the compressed DataFrame
output_path = f"../data/{VACCINE}/RNA_circos_compressed_by_feature.csv"
compressed_data.to_csv(output_path, index=False)
print(f"Compressed data saved to {output_path}")