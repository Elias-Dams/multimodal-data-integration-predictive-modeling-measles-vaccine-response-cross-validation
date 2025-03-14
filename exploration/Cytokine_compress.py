import pandas as pd
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read the cytokine data (assumes rows are subjects and columns are cytokine measurements)
cytokine_data = pd.read_csv("../data/Measles/cytokines_data.csv", index_col=0)

# Read the clusters from the JSON file
with open("../data/Measles/clusters/cytokines.json", "r") as f:
    clusters = json.load(f)

# Create an empty DataFrame to store the module scores, using the same index as cytokine_data
module_scores = pd.DataFrame(index=cytokine_data.index)

# Loop through each cluster to compute module scores via PCA
for cluster_name, genes in clusters.items():
    # Identify the genes present in the cytokine data
    common_genes = [gene for gene in genes if gene in cytokine_data.columns]
    if not common_genes:
        print(f"No overlapping genes found for {cluster_name}. Skipping this cluster.")
        continue

    # Subset the cytokine data to only include the genes in this cluster
    cluster_data = cytokine_data[common_genes]

    # Standardize the data (zero mean and unit variance)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(cluster_data)

    # Perform PCA to reduce the data to one component
    pca = PCA(n_components=1)
    pca_scores = pca.fit_transform(standardized_data)

    # Flatten the scores and store them in the module_scores DataFrame under the cluster name
    module_scores[cluster_name] = pca_scores.flatten()

# Write the resulting module scores to a CSV file
module_scores.to_csv("../data/Measles/cytokine_modules.csv")
print("Module scores saved to '../data/Measles/cytokine_modules.csv'")