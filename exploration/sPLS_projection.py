
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression

from exploration.SMOTE_utils import handle_missing_values, load_groups_from_json, compress_correlated_features, \
    encode_labels
from exploration.load_datasets import get_measles_data, get_hepatitis_data
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.patches import Circle, Ellipse
from functools import reduce
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np

COMPRESS_CORRELATED = False
CENTROID_SCALE = 4

###########################################################################
# LOAD Measles THE DATA
###########################################################################

datasets_merged_measles, abtiters_measles = get_measles_data(visualise=False, add_metadate=True)

for key in ["cytokines", "clonal_breadth", "clonal_depth", "RNa_data", "Meta"]:
    datasets_merged_measles.pop(key, None)
datasets_merged_measles['cytometry']['df'] = datasets_merged_measles['cytometry']['df'][['Vaccinee', 'WBC Day 0', '%GRA Day 0', '%LYM Day 0']]

print(datasets_merged_measles.keys())

###########################################################################
# HANDLE MISSING VALUES
###########################################################################

for dataset in datasets_merged_measles:
    datasets_merged_measles[dataset]["df"] = handle_missing_values(datasets_merged_measles[dataset]["df"],dataset, abtiters_measles, strategy='mean')

###########################################################################
# COMPRESS CORRELATED FEATURES
###########################################################################
if COMPRESS_CORRELATED:
    cytokines_groups = load_groups_from_json(f"../data/Measles/clusters/cytokines.json")
    cytometry_groups = load_groups_from_json(f"../data/Measles/clusters/cytometry.json")
    rna_groups = load_groups_from_json(f"../data/Combined/clusters/RNA1.json")

    if "cytokines" in datasets_merged_measles:
        datasets_merged_measles['cytokines']["df"] = compress_correlated_features(datasets_merged_measles['cytokines']["df"], cytokines_groups)
    if "cytometry" in datasets_merged_measles:
        datasets_merged_measles['cytometry']["df"] = compress_correlated_features(datasets_merged_measles['cytometry']["df"], cytometry_groups)
    if "RNa_data" in datasets_merged_measles:
        datasets_merged_measles['RNa_data']["df"] = compress_correlated_features(datasets_merged_measles['RNa_data']["df"], rna_groups)



###########################################################################
# SCALE THE FEATURES INDIVIDUALLY
###########################################################################
# When working with multiple datasets measured on different scales , it’s
# best to normalize each dataset individually before concatenating them.
# This approach ensures that each set of features is scaled appropriately
# relative to its own distribution, so that no single dataset dominates
# due to differences in units or variability.

for dataset in datasets_merged_measles:
    df = datasets_merged_measles[dataset]["df"]
    # Identify numeric columns (skip non-numeric ones like IDs if present)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # If there are numeric columns, scale them
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save the normalized DataFrame back into the dictionary
    datasets_merged_measles[dataset]["df"] = df

###########################################################################
# MERGE TO 1 BIG DATASET
###########################################################################

# Define columns to drop that are common across datasets
common_drop_cols = ["response_label"]

# Create a list of DataFrames, dropping the duplicate common columns
dfs = []
for ds in datasets_merged_measles:
    df = datasets_merged_measles[ds]["df"].copy()
    # Drop any common duplicate columns if they exist
    df = df.drop(columns=[col for col in common_drop_cols if col in df.columns], errors='ignore')
    dfs.append(df)

# Merge all DataFrames on the "Vaccinee" column using an outer join
merged_df_measles = reduce(lambda left, right: pd.merge(left, right, on="Vaccinee", how="outer"), dfs)

# Use abtiters to get the response labels back
labels_df_measles = abtiters_measles[["Vaccinee", "response_label"]].copy()
merged_df_measles = pd.merge(merged_df_measles, labels_df_measles, on="Vaccinee", how="left")

print("Merged dataset:")
print(merged_df_measles.head())

###########################################################################
# sPLS PROJECTION
###########################################################################

def _add_reference_circles(ax, radii=[0.5, 1.0], color='gray', linestyle='--', linewidth=0.8, alpha=0.5):
    for radius in radii:
        circle = plt.Circle((0, 0), radius, color=color, fill=False, linestyle=linestyle, linewidth=linewidth,
                            alpha=alpha)
        ax.add_artist(circle)

def scale_to_range(arr, new_min=-1, new_max=1):
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        return np.full_like(arr, new_min)
    scaled = (arr - arr_min) / (arr_max - arr_min) * (new_max - new_min) + new_min
    return scaled

def compute_radius(subdf, centroid):
    distances = np.sqrt(
        (subdf['Component1'] - centroid['Component1']) ** 2 + (subdf['Component2'] - centroid['Component2']) ** 2)
    # Optionally, use a percentile (say, 90th) instead of max to avoid extreme outliers
    radius = np.percentile(distances, 90)
    return radius

def adjust_sign(data):
    """
    If the median of the data is negative, flip the sign.
    Otherwise, return data unchanged.
    """
    if np.median(data) < 0:
        return -data
    else:
        return data

def plot_pls_with_response_marginals(
        data,
        response_label_col='response_label',
        centroid_scale = 2,
        percentile=90,
        n_components=2,
        centroid_info = None,
        pls = None,
        verbose=False,
        title = "Combined PLS Scatter Plot with Marginals"

):
    """
    Plots a PLS scatter plot with marginal boxplots grouped by response labels,
    and includes a biplot with feature contributions based on loadings.
    """
    # Select numeric features for PLS
    numeric_features = data.select_dtypes(include=[float, int]).columns.tolist()

    # Encode the target variable if it's categorical
    if data[response_label_col].dtype == 'object' or data[response_label_col].dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(data[response_label_col])
        print(dict(zip(le.classes_, le.transform(le.classes_))))
    else:
        y = data[response_label_col].values

    X = data[numeric_features].values

    # Perform PLS Regression
    if pls is None:
        pls = PLSRegression(n_components=n_components)
        pls.fit(X, y)

    # Transform the data
    X_pls = pls.transform(X)
    # data[f'Component1'] = X_pls[:, 0]
    # data[f'Component2'] = X_pls[:, 1]

    data['Component1'] = adjust_sign(scale_to_range(X_pls[:, 0], new_min=-2, new_max=2))
    data['Component2'] = adjust_sign(scale_to_range(X_pls[:, 1], new_min=-2, new_max=2))

    # Extract Y scores (latent variables for Y)
    Y_pls = pls.y_scores_

    # Calculate Pearson correlation for each component pair
    corr1, _ = pearsonr(pls.x_scores_[:, 0], Y_pls[:, 0])
    corr2, _ = pearsonr(pls.x_scores_[:, 1], Y_pls[:, 1])

    print(f"X-Y Variates 1 Correlation: {corr1:.2f}")
    print(f"X-Y Variates 2 Correlation: {corr2:.2f}")

    # Extract and scale loadings
    loadings = pls.x_weights_
    norms = np.linalg.norm(loadings, axis=1)
    max_norm = norms.max()
    scaled_loadings = loadings / max_norm

    # Feature selection based on percentile for Component1 and Component2
    loadings_component1 = np.abs(pls.x_weights_[:, 0])
    threshold_component1 = np.percentile(loadings_component1, percentile)
    top_features_component1 = data[numeric_features].columns[loadings_component1 >= threshold_component1]

    loadings_component2 = np.abs(pls.x_weights_[:, 1])
    threshold_component2 = np.percentile(loadings_component2, percentile)
    top_features_component2 = data[numeric_features].columns[loadings_component2 >= threshold_component2]

    if verbose:
        print(f"\nTop Features Contributing to Component1 (Top {100 - percentile}%):")
        print(top_features_component1.tolist())
        print(f"Loading threshold at the {percentile}th percentile: {threshold_component1:.4f}")

        print(f"\nTop Features Contributing to Component2 (Top {100 - percentile}%):")
        print(top_features_component2.tolist())
        print(f"Loading threshold at the {percentile}th percentile: {threshold_component2:.4f}")

    # Create the plot
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7), wspace=0.10, hspace=0.10)
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    fig.suptitle(title, fontsize=24)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.94, bottom=0.07)

    # Scatter plot grouped by response labels
    scatter_kwargs = {
        'data': data,
        'x': 'Component1',
        'y': 'Component2',
        'ax': ax_scatter,
        'hue': response_label_col,
        's': 100
    }
    sns.scatterplot(**scatter_kwargs)

    grouped = data.groupby('response_label')[['Component1', 'Component2']]

    # For each group, plot an ellipse
    centroid_info_return = {}

    for label, group in grouped:
        if centroid_info:
            info = centroid_info[label]
            centroid = info['centroid']
            width = info['width']
            height = info['height']
            angle = info['angle']
        else:
            # Calculate from group data
            coords = group[['Component1', 'Component2']].values
            centroid = np.mean(coords, axis=0)
            cov = np.cov(coords, rowvar=False)
            eigenvals, eigenvecs = np.linalg.eig(cov)
            # Sort eigenvalues (and eigenvectors) in descending order
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width = centroid_scale * np.sqrt(eigenvals[0])
            height = centroid_scale * np.sqrt(eigenvals[1])

            centroid_info_return[label] = {
                "centroid": centroid,
                "angle": angle,
                "width": width,
                "height": height
            }

        ellipse = Ellipse(xy=centroid, width=width, height=height, angle=angle, edgecolor='black', facecolor='none', linewidth=1, linestyle='--')
        ax_scatter.add_patch(ellipse)
        ax_scatter.text(centroid[0] + 0.05, centroid[1] + 0.05, label, fontsize=12, fontweight='bold', color='black')

    ax_scatter.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_scatter.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    # Marginal boxplots grouped by response labels
    sns.boxplot(
        data=data,
        x='Component1',
        y=response_label_col,
        orient='h',
        ax=ax_hist_x,
        hue=response_label_col,
        dodge=False
    )

    sns.boxplot(
        data=data,
        x=response_label_col,
        y='Component2',
        orient='v',
        ax=ax_hist_y,
        hue=response_label_col,
        dodge=False
    )

    # Hide labels on marginal plots for cleaner visualization
    ax_hist_x.tick_params(axis='x', labelbottom=False)
    ax_hist_y.tick_params(axis='y', labelleft=False)

    # Axis labels and title
    ax_scatter.set_xlabel(f"X-Y Variates 1 Correlation: {corr1 * 100:.0f}%")
    ax_scatter.set_ylabel(f"X-Y Variates 2 Correlation: {corr2 * 100:.0f}%")
    ax_scatter.set_title("PLS Scatter Plot with Feature Biplot")
    plt.show()

    return centroid_info_return, pls

centroid_info_measles, pls_measles = plot_pls_with_response_marginals(
    data=merged_df_measles.copy(),
    response_label_col='response_label',
    centroid_scale = CENTROID_SCALE,
    verbose=True,
    title = "Combined PLS Scatter Plot with Marginals (Measles)"
)

###########################################################################
# LOAD hepatitis B DATA
###########################################################################

datasets_merged_hepatitis, abtiters_hepatitis = get_hepatitis_data(visualise=False, add_metadate=True)

for key in ["TCR_predictions", "RNa_data", "Meta"]:
    datasets_merged_hepatitis.pop(key, None)

datasets_merged_hepatitis['cytometry']['df'] = datasets_merged_hepatitis['cytometry']['df'][['Vaccinee', 'WBC Day 0', '%GRA Day 0', '%LYM Day 0']]
print(datasets_merged_hepatitis.keys())

###########################################################################
# HANDLE MISSING VALUES
###########################################################################

for dataset in datasets_merged_hepatitis:
    datasets_merged_hepatitis[dataset]["df"] = handle_missing_values(datasets_merged_hepatitis[dataset]["df"],dataset, abtiters_hepatitis, strategy='mean')

###########################################################################
# COMPRESS CORRELATED FEATURES
###########################################################################
if COMPRESS_CORRELATED:
    cytometry_groups = load_groups_from_json("../data/Hepatitis B/clusters/cytometry.json")
    rna_groups = load_groups_from_json("../data/Combined/clusters/RNA1.json")

    if 'cytometry' in datasets_merged_hepatitis:
        datasets_merged_hepatitis['cytometry']["df"] = compress_correlated_features(datasets_merged_hepatitis['cytometry']["df"],cytometry_groups)
    if 'RNa_data' in  datasets_merged_hepatitis:
        datasets_merged_hepatitis['RNa_data']["df"] = compress_correlated_features(datasets_merged_hepatitis['RNa_data']["df"], rna_groups)

###########################################################################
# SCALE THE FEATURES INDIVIDUALLY
###########################################################################

for dataset in datasets_merged_hepatitis:
    df = datasets_merged_hepatitis[dataset]["df"]
    # Identify numeric columns (skip non-numeric ones like IDs if present)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # If there are numeric columns, scale them
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save the normalized DataFrame back into the dictionary
    datasets_merged_hepatitis[dataset]["df"] = df

###########################################################################
# MERGE TO 1 BIG DATASET
###########################################################################

# Define columns to drop that are common across datasets
common_drop_cols = ["response_label"]

# Create a list of DataFrames, dropping the duplicate common columns
dfs = []
for ds in datasets_merged_hepatitis:
    df = datasets_merged_hepatitis[ds]["df"].copy()
    # Drop any common duplicate columns if they exist
    df = df.drop(columns=[col for col in common_drop_cols if col in df.columns], errors='ignore')
    dfs.append(df)

# Merge all DataFrames on the "Vaccinee" column using an outer join
merged_df_hepatitis = reduce(lambda left, right: pd.merge(left, right, on="Vaccinee", how="outer"), dfs)

# Use abtiters to get the response labels back
labels_df_hepatitis = abtiters_hepatitis[["Vaccinee", "response_label"]].copy()
merged_df_hepatitis = pd.merge(merged_df_hepatitis, labels_df_hepatitis, on="Vaccinee", how="left")

print("Merged dataset:")
print(merged_df_hepatitis.head())

###########################################################################
# sPLS PROJECTION
###########################################################################

plot_pls_with_response_marginals(
    data=merged_df_hepatitis.copy(),
    response_label_col='response_label',
    centroid_scale = CENTROID_SCALE,
    centroid_info=centroid_info_measles,
    verbose=True,
    title = "Combined PLS Scatter Plot with Marginals (Hepatitis B)"
)
