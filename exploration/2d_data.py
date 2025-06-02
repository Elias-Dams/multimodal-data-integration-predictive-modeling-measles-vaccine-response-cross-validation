import random

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from helper_functions.load_datasets import get_measles_data


def plot_pca_with_labels(data_df: pd.DataFrame, dataset_name: str):
    if data_df.shape[1] < 2:
        print(f"Skipping PCA for {dataset_name}: Data has fewer than 2 features.")
        return

    sample_labels = data_df['Vaccinee']  # Column for point labels (Vaccinee number)
    color_labels = data_df['response_label']  # Column for point colors (Response label)
    data_df_cleaned = data_df.drop(columns=['Vaccinee', 'response_label'])

    scaler = StandardScaler()
    data_cleaned_scaled = scaler.fit_transform(data_df_cleaned)
    data_df_cleaned_scaled = pd.DataFrame(data_cleaned_scaled, index=data_df_cleaned.index, columns=data_df_cleaned.columns)

    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(data_df_cleaned_scaled)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(data=principal_components,
                          columns=['Principal Component 1', 'Principal Component 2'],
                          index=data_df_cleaned_scaled.index)

    # Add sample names to the PCA DataFrame
    pca_df['Sample'] = sample_labels
    pca_df['Response'] = color_labels

    # Create the plot using seaborn for coloring
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Response', data=pca_df)

    # Add text labels for each point (using the 'Sample' column which contains Vaccinee numbers)
    for i in range(pca_df.shape[0]):
        plt.text(pca_df['Principal Component 1'].iloc[i],
                 pca_df['Principal Component 2'].iloc[i],
                 pca_df['Sample'].iloc[i],
                 fontsize=9,
                 ha='right')  # Horizontal alignment

    plt.title(f'PCA of {dataset_name} Data with Sample Labels and Response Coloring')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.grid(True)
    plt.show()

datasets_merged, abtiters = get_measles_data()

# Loop through each dataset and plot PCA
for dataset in datasets_merged.items():
    print(f"Generating PCA plot for: {dataset[0]}")
    plot_pca_with_labels(dataset[1][f'df'], dataset[0])
