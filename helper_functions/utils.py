import random

from sklearn.preprocessing import StandardScaler
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import matplotlib.colors as mcolors
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from sklearn.utils import resample, shuffle
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import numpy as np
import pandas as pd
import joblib
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def check_missing_values(df, dataset_name, nr_samples):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print(f"{dataset_name}:")

    if len(df['Vaccinee']) < nr_samples:
        print(f"\tMissing samples: only contains {len(df['Vaccinee'])} samples instead of {nr_samples}.")

    if not missing.empty:
        print(f"\tMissing values:")
        print(f"\t{missing}")
    else:
        print(f"\tNo missing values.")

def handle_missing_values(df, dataset_name, complete_samples, strategy='mean'):
    print(f"Handling missing values for {dataset_name}...")
    new_df = df.copy()

    # Identify missing samples
    missing_samples = complete_samples[~complete_samples['Vaccinee'].isin(df['Vaccinee'])]
    if missing_samples.empty:
        print(f"\tNo missing samples to add for {dataset_name}.")
        return new_df

    print(f"\tFound {len(missing_samples)} missing samples to process.")

    # Function to prepare training subset
    def prepare_training_subset(label, df):
        if pd.notnull(label):
            subset = df[df['response_label'] == label].drop(['Vaccinee', 'response_label'], axis=1)
            return subset if not subset.empty else df.drop(['Vaccinee', 'response_label'], axis=1)
        return df.drop(['Vaccinee', 'response_label'], axis=1)

    # Imputation logic
    imputed_features_list = []
    imputer_class = KNNImputer if strategy.lower() == 'knn' else SimpleImputer
    imputer_kwargs = {'strategy': strategy} if strategy.lower() != 'knn' else {'n_neighbors': 3}

    for label, group in missing_samples.groupby('response_label'):
        # Prepare training subset
        train_subset = prepare_training_subset(label, df)

        # Ensure all columns in train_subset exist in the group
        features = group[['Vaccinee', 'response_label']].copy()  # Keep 'Vaccinee' and 'response_label' intact
        for col in train_subset.columns:
            if col not in features.columns:
                features[col] = None  # Add missing columns with None

        # Drop 'Vaccinee' and 'response_label' for imputation
        imputation_input = features.drop(['Vaccinee', 'response_label'], axis=1)

        # Impute missing values
        imputer = imputer_class(**imputer_kwargs)
        imputer.fit(train_subset)
        imputed_values = imputer.transform(imputation_input)

        # Add imputed values back to the features DataFrame
        for i, col in enumerate(imputation_input.columns):
            features[col] = imputed_values[:, i]

        imputed_features_list.append(features)

    # Combine all imputed groups
    imputed_features = pd.concat(imputed_features_list, axis=0).reset_index(drop=True)

    # Append imputed samples to the original dataframe
    new_df = pd.concat([df, imputed_features], ignore_index=True)
    new_df = new_df.set_index('Vaccinee').reindex(complete_samples['Vaccinee']).reset_index()
    print(f"\tAdded {len(imputed_features)} samples to {dataset_name}.")

    return new_df

def compress_correlated_features(data, groups):
    """
    Compress groups of highly correlated features into a single principal component for each group.

    Parameters:
    - data (pd.DataFrame): The input dataframe containing the features.
    - groups (dict): A dictionary where keys are group names and values are lists of correlated features.

    Returns:
    - data (pd.DataFrame): The dataframe with compressed features added and original features removed.
    """
    skipped = []

    def compress_features(data, features, new_feature_name):
        for f in features:
            if f not in data.columns:
                skipped.extend(features)
                print(
                    f"Warning: Feature '{f}' not found in DataFrame. Skipping compression for '{new_feature_name}' as at least one feature is missing.")
                return data.copy()  # Return original data (or a copy) if any feature is missing

        # Standardize the features
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(data[features])

        # Perform PCA to extract the first principal component
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(standardized_features)

        # Add the new compressed feature to the dataframe
        data[new_feature_name] = principal_component
        return data

    # Iterate over each group and compress the features
    for group_name, features in groups.items():
        new_feature_name = f"{group_name}_Compressed"
        data = compress_features(data, features, new_feature_name)

    # Drop the original features that were compressed
    all_features_to_drop = [feature for group in groups.values() for feature in group if feature not in skipped]
    data.drop(columns=all_features_to_drop, inplace=True)

    return data

def load_groups_from_json(file_path):
    with open(file_path, "r") as file:
        groups = json.load(file)
    return groups

def encode_labels(df):
    le = LabelEncoder()
    df['response_label'] = le.fit_transform(df['response_label'])
    mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print("Label mapping:", mapping)
    return df

def visualize_pca_oversampling_with_label(
        X_train, y_train,
        X_train_resampled, y_train_resampled,
        label_colors={0: "blue", 1: "green", 2: "red"},
        data_name = None, model_name = None, methode_name = None
):
    """
    Perform a 2D PCA on the resampled training data, coloring points by their label,
    drawing a red circle (edgecolor='red') around synthetic (new) samples,
    and marking removed samples from the original training set with an "x".
    """

    def row_to_tuple(row):
        return tuple(np.round(row, decimals=5))

    # Create sets of rounded rows
    original_set = {row_to_tuple(row) for row in X_train.to_numpy()}
    resampled_set = {row_to_tuple(row) for row in X_train_resampled.to_numpy()}

    # Label resampled set: "org" if from original, "new" if synthetic.
    types_resampled = ["org" if row_to_tuple(row) in original_set else "new"
                       for row in X_train_resampled.to_numpy()]

    # Label original set: "org" if still present in resampled, "del" if removed.
    types_train = ["org" if row_to_tuple(row) in resampled_set else "del"
                   for row in X_train.to_numpy()]

    # Save labels in dataframes for later use
    X_train_labeled = X_train.copy()
    X_train_labeled["type"] = types_train

    X_train_resampled_labeled = X_train_resampled.copy()
    X_train_resampled_labeled["type"] = types_resampled

    scaler = StandardScaler()
    # Fit and transform the training data, and transform the test data
    X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
    X_train_scaled = scaler.transform(X_train)
    # Wrap the numpy arrays back into DataFrames with the original indices and column names
    X_train_resampled_scaled = pd.DataFrame(X_train_resampled_scaled, index=X_train_resampled.index,columns=X_train_resampled.columns)
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

    # Fit PCA on the resampled scaled data
    pca = PCA(n_components=2, random_state=42)
    pca_coords_resampled = pca.fit_transform(X_train_resampled_scaled)

    # Transform the original scaled data into the same PCA space
    pca_coords_original = pca.transform(X_train_scaled)

    # Create PCA dataframes with corresponding labels (indexes assumed aligned)
    df_resampled_pca = pd.DataFrame(pca_coords_resampled, columns=["PC1", "PC2"], index=X_train_resampled.index)
    df_resampled_pca["type"] = X_train_resampled_labeled["type"]
    df_resampled_pca["label"] = y_train_resampled

    df_original_pca = pd.DataFrame(pca_coords_original, columns=["PC1", "PC2"], index=X_train.index)
    df_original_pca["type"] = X_train_labeled["type"]
    df_original_pca["label"] = y_train

    # Select only the 'del' samples from the original PCA dataframe
    df_original_del = df_original_pca[df_original_pca["type"] == "del"]

    # Concatenate the resampled PCA dataframe with the deleted samples
    df_union_pca = pd.concat([df_resampled_pca, df_original_del], axis=0)

    # Plotting the union points with markers for synthetic (new) and removed samples.
    plt.figure(figsize=(8, 6))
    for idx, row in df_union_pca.iterrows():
        pc1 = row["PC1"]
        pc2 = row["PC2"]
        sample_type = row["type"]
        color = label_colors[row["label"]]

        # Apply plotting logic based on type
        if sample_type == "new":
            # Plot new synthetic samples with a red circle edge
            plt.scatter(pc1, pc2, color=color, edgecolor="red",
                        linewidth=1.2, s=40, alpha=0.8)
        elif sample_type == "del":
            # Plot deleted samples with an 'x' marker
            plt.scatter(pc1, pc2, color="grey", marker="x", linewidth=1.2,
                        s=40, alpha=0.8)
        else:
            # Plot original samples normally (using the green color)
            plt.scatter(pc1, pc2, color=color, edgecolor="none",
                        s=40, alpha=0.8)

    plt.title(f"Train Set \n(Data: {data_name} Model: {model_name}, Methode: {methode_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle=":", alpha=0.4)

    label_names = {
        0: "Non responder",
        1: "responder",
        2: "created sample"
    }

    # Create a legend for the original labels (avoid duplicate "Removed" entries)
    handles = {}
    for lbl_val, lbl_color in label_colors.items():
        if lbl_val not in handles:
            handles[lbl_val] = plt.Line2D([0], [0], marker='o', color=lbl_color,
                                          label=f"{label_names[lbl_val]}",
                                          markerfacecolor=lbl_color, markersize=8,
                                          linewidth=0)
    # Add one legend entry for removed samples.
    handles["Removed"] = plt.Line2D([0], [0], marker='x', linestyle='None', color='black',
                                    label="Removed", markersize=8)
    plt.legend(handles=list(handles.values()), title="Labels")
    plt.show()

    return pca

def visualize_decision_boundary_in_pca_space(model, pca, X_train, y_train, X_test, y_test, n_points=500 ,
        data_name = None, model_name = None, methode_name = None, save=False):
    """
    Visualize the decision boundary in PCA space for both training and test sets in separate plots.
    Both plots use the same grid so that the decision boundary is consistent.
    """
    # Transform both training and test sets into PCA space
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Compute overall grid boundaries using both train and test data
    all_pca = np.vstack([X_train_pca, X_test_pca])
    x_min, x_max = all_pca[:, 0].min(), all_pca[:, 0].max()
    y_min, y_max = all_pca[:, 1].min(), all_pca[:, 1].max()

    # Add margins (10% of the range)
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_margin, x_max + x_margin, n_points),
        np.linspace(y_min - y_margin, y_max + y_margin, n_points)
    )
    grid_pca = np.c_[xx.ravel(), yy.ravel()]

    # Inverse-transform the grid to the original feature space for predictions
    grid_original = pca.inverse_transform(grid_pca)
    Z = model.predict(grid_original).reshape(xx.shape)

    # Create a custom colormap (blue-green, for example)
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_green", ["blue", "green"], N=256)

    # -------------------
    # Plot for Training Data
    # -------------------
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.linspace(Z.min(), Z.max(), 100))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap, edgecolor="k", s=50)
    plt.title(f"Decision Boundary (Train) \n(Data: {data_name} Model: {model_name}, Methode: {methode_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

    # -------------------
    # Plot for Test Data
    # -------------------
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.linspace(Z.min(), Z.max(), 100))
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap, edgecolor="k", s=50)
    plt.title(f"Decision Boundary (Test) \n(Data: {data_name} Model: {model_name}, Methode: {methode_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

    if save:
        joblib.dump(pca, f'../data_created/SMOTE/saved_decision_boundaries/pca_{data_name}_{model_name}_{methode_name}.pkl')
        joblib.dump(model, f'../data_created/SMOTE/saved_decision_boundaries/trained_{data_name}_{model_name}_{methode_name}.pkl')


def custom_train_test_split(data, labels, test_size=0.2, random_state=42):
    # Ensure 'data' and 'labels' have the same index
    if not data.index.equals(labels.index):
        raise ValueError("'data' and 'labels' must have the same index.")

    unique_classes = np.unique(labels)
    if len(unique_classes) != 2:
        raise ValueError("This function currently only supports binary classification based on the labels.")

    # Calculate the total number of samples for the test set based on the proportion
    total_test_samples = int(len(data) * test_size)


    # Calculate the number of samples needed PER CLASS for the test set
    base_test_size_per_class = total_test_samples // 2

    test_size_class_0 = base_test_size_per_class
    test_size_class_1 = base_test_size_per_class

    # If the total number of test samples is odd, add one extra to the larger class
    if total_test_samples % 2 != 0:
        # Get the counts of each class to determine the larger one
        class_counts = labels.value_counts()
        if class_counts[unique_classes[0]] > class_counts[unique_classes[1]]:
            test_size_class_0 += 1  # Add one to the count for class 0
        else:
            test_size_class_1 += 1  # Add one to the count for class 1

        # Separate indices by class based on the labels Series
    indices_class_0 = labels[labels == unique_classes[0]].index
    indices_class_1 = labels[labels == unique_classes[1]].index

    # Check if the calculated test_size_per_class is feasible given the class sizes
    # --- Check if the calculated test sizes per class are feasible ---
    if test_size_class_0 > len(indices_class_0):
        raise ValueError(
            f"Calculated test size for class {unique_classes[0]} ({test_size_class_0}) "
            f"is greater than the number of available samples ({len(indices_class_0)})."
        )
    if test_size_class_1 > len(indices_class_1):
        raise ValueError(
            f"Calculated test size for class {unique_classes[1]} ({test_size_class_1}) "
            f"is greater than the number of available samples ({len(indices_class_1)})."
        )

    # Randomly sample 'test_size_per_class' indices from each class for the test set
    rng = np.random.RandomState(random_state)  # Use RandomState for consistent random numbers
    test_indices_class_0 = rng.choice(indices_class_0, size=test_size_class_0, replace=False)
    test_indices_class_1 = rng.choice(indices_class_1, size=test_size_class_1, replace=False)

    # Combine the indices selected for the test set
    test_indices = np.concatenate([test_indices_class_0, test_indices_class_1])

    # The training indices are all original indices that were NOT selected for the test set
    train_indices = data.index.difference(test_indices)

    # --- Split the input 'data' Series using the determined indices ---
    train_data = data.loc[train_indices]
    test_data = data.loc[test_indices]

    return train_data, test_data


def train_val_test_split_exception(data, labels, exceptions_data, exceptions_labels, random_state=42):
    """
    Splits data into train, validation, and test sets, ensuring the test set has
    1 positive and 5 negative samples from the exception set.

    Returns:
    - train_data, val_data, train_data_exceptions, val_data_exceptions, test_data, test_labels
    """
    random.seed(random_state)
    np.random.seed(random_state)

    # Step 1: Build test set from exceptions_data (1 pos, 5 neg)
    pos_indices = [i for i, l in enumerate(exceptions_labels) if l == 1]
    neg_indices = [i for i, l in enumerate(exceptions_labels) if l == 0]

    test_pos_idx = random.sample(pos_indices, 1)
    test_neg_idx = random.sample(neg_indices, 5)
    test_indices = test_pos_idx + test_neg_idx

    test_data = [exceptions_data[i] for i in test_indices]

    # Remove test samples from data and labels
    test_ids = set(test_data)
    full_data_filtered = [(d, l) for d, l in zip(data, labels) if d not in test_ids]
    exceptions_filtered = [(d, l) for d, l in zip(exceptions_data, exceptions_labels) if d not in test_ids]

    # Unzip data and labels
    data_rest, labels_rest = zip(*full_data_filtered)
    exceptions_rest, exceptions_labels_rest = zip(*exceptions_filtered)

    # Create Series indexed by Vaccinee ID
    test_data_series = pd.Series(test_data, index=test_data, name="Vaccinee")
    data_rest_series = pd.Series(data_rest, index=data_rest, name="Vaccinee")
    labels_rest_series = pd.Series(labels_rest, index=data_rest, name="response_label")
    exceptions_rest_series = pd.Series(exceptions_rest, index=exceptions_rest, name="Vaccinee")
    exceptions_labels_series = pd.Series(exceptions_labels_rest, index=exceptions_rest, name="response_label")

    # Step 2: Split remaining main data (82.4% train, 17.6% val)
    train_data, val_data = train_test_split(
        data_rest_series,  train_size=0.824, random_state=random_state, stratify=labels_rest_series
    )

    # Step 3: Split remaining exceptions (81% train, 19% val)
    train_data_exceptions, val_data_exceptions = train_test_split(
        exceptions_rest_series, train_size=0.81, random_state=random_state, stratify=exceptions_labels_series
    )

    np.random.default_rng()

    return (
        train_data, val_data,
        train_data_exceptions, val_data_exceptions,
        test_data_series,
    )

def oversample_data(X_train, y_train, oversampling_method, random_state=42):

    min_class_count = min(Counter(y_train).values())

    if oversampling_method == 'smote':
        # Perform oversampling on the training data
        sm = SMOTE(random_state=random_state, k_neighbors=min(min_class_count-1, 5))
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
        # Creates new synthetic examples by interpolating between a minority class instance and one of its k-nearest neighbors.
        # This helps to enlarge the minority class without simply duplicating existing samples.

    elif oversampling_method == 'smote-borderline':
        # Perform oversampling on the training data
        sm = BorderlineSMOTE(kind='borderline-1', random_state=random_state, k_neighbors=min(min_class_count-1, 5))
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
        # Focuses on instances that lie near the decision boundary (the “borderline” samples).
        # By generating new examples only for those hard-to-learn cases, it aims to improve the classifier’s ability to separate classes where they overlap.

    elif oversampling_method == 'smote-adasyn':
        # Perform oversampling on the training data
        sm = ADASYN(random_state=random_state, n_neighbors=min(min_class_count-1, 5))
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
        # Similar to SMOTE, but it adaptively generates more synthetic examples for minority class samples that are harder to learn
        # (i.e., those with fewer minority neighbors). This way, the algorithm shifts the decision boundary toward the
        # majority class and focuses on areas with higher complexity.

    elif oversampling_method == 'smote-smotetomek':
        # Perform oversampling on the training data
        sm = SMOTETomek(smote=SMOTE(random_state=random_state, k_neighbors=min(min_class_count-1, 5)), random_state=random_state)
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
        # Combines SMOTE with an under-sampling technique called Edited Nearest Neighbors (ENN).
        # After oversampling with SMOTE, ENN removes samples that are misclassified by their neighbors.
        # This helps clean up noise and can improve model performance.

    elif oversampling_method == 'smote-smoteenn':
        # Perform oversampling on the training data
        sm = SMOTEENN(smote=SMOTE(random_state=random_state, k_neighbors=min(min_class_count-1, 5)), random_state=random_state)
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
        # Merges SMOTE with Tomek links under-sampling. Tomek links are pairs of samples from opposite classes that are each other’s nearest neighbors.
        # Removing these links after SMOTE oversampling can help eliminate overlapping samples and create a cleaner decision boundary.

    elif oversampling_method == 'random':
        # Perform random oversampling on the training data (simply duplicates existing minority class samples)
        ros = RandomOverSampler(random_state=random_state)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    else:
        raise ValueError(f"Unsupported oversampling method: {oversampling_method}")

    # After oversampling
    print(f"Resampled class distribution using {oversampling_method} for train set: {Counter(y_train_resampled)}")

    # Check if oversampling left only one class
    if len(np.unique(y_train_resampled)) < 2:
        print(f"Only one class present after oversampling with {oversampling_method}. Skipping this dataset.")
        return None, None

    class_counts = Counter(y_train_resampled)
    if min(class_counts.values()) < 5:
        print(
            f"Error: n_splits = {5} is greater than the number of samples in at least one class in the training set: {class_counts}")
        return None, None

    return X_train_resampled, y_train_resampled

def split_dataset(df, train_vaccinees, test_vaccinees, oversampling_method=None, visualise_oversampling=False, data_name = None, model_name = None, methode_name = None, random_state = 42):
    # Use the pre-defined train and test vaccinees
    train_df = df[df['Vaccinee'].isin(train_vaccinees)]
    test_df = df[df['Vaccinee'].isin(test_vaccinees)]

    # Separate features and labels
    X_train = train_df.drop(['Vaccinee', 'response_label'], axis=1)
    y_train = train_df['response_label']
    X_test = test_df.drop(['Vaccinee', 'response_label'], axis=1)
    y_test = test_df['response_label']

    X_train_resampled = X_train
    y_train_resampled = y_train

    pca = None

    if oversampling_method is not None:
        X_train_resampled, y_train_resampled = oversample_data(X_train, y_train, oversampling_method, random_state=random_state)

        if X_train_resampled is None or y_train_resampled is None:
            return None, None, None, None, None

        if visualise_oversampling:
            pca = visualize_pca_oversampling_with_label(
                X_train, y_train,
                X_train_resampled,
                y_train_resampled,
                data_name=data_name,
                model_name=model_name,
                methode_name=methode_name
            )

    else:
        # No oversampling
        print(f"Class distribution for train set: {Counter(y_train_resampled)}")
        print(f"Class distribution for test set: {Counter(y_test)}")

    return X_train_resampled, X_test, y_train_resampled, y_test, pca

def scale_features(X_train, X_test, X_val = None):
    scaler = StandardScaler()
    # Fit and transform the training data, and transform the test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)

    # Wrap the numpy arrays back into DataFrames with the original indices and column names
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    if X_val is not None:
        X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)

    if X_val is not None:
        return X_train_scaled, X_test_scaled, X_val_scaled
    else:
        return X_train_scaled, X_test_scaled

def plot_boot_scores_with_ci(boot_scores, confidence_level=0.95):
    """
    Plots the distribution of boot_scores with confidence interval (CI).

    Parameters:
    - boot_scores: List or array of bootstrapped scores.
    - confidence_level: The desired confidence level for the CI (default is 0.95).

    Returns:
    - Lower and upper bounds of the confidence interval.
    """
    # Calculate the percentiles for the confidence interval
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    lower_bound = np.percentile(boot_scores, lower_percentile)
    upper_bound = np.percentile(boot_scores, upper_percentile)

    # Plotting the distribution of boot_scores
    plt.hist(boot_scores, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label=f'Lower CI ({lower_bound:.3f})')
    plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=2, label=f'Upper CI ({upper_bound:.3f})')

    plt.title('Distribution of Boot Scores with Confidence Interval')
    plt.xlabel('Boot Scores')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Return the confidence interval bounds
    return lower_bound, upper_bound

def print_permutation_results(dataset_name, p_value, observed_accuracy, permuted_scores):
    print(f"\n--- {dataset_name} Permutation Test Results ---")
    print(f"\tObserved Accuracy: {observed_accuracy:.4f}")
    print(f"\tP-Value: {p_value:.4f}")

    # Summary of permuted scores
    permuted_mean = np.mean(permuted_scores)
    permuted_std = np.std(permuted_scores)
    print(f"\tPermuted Scores Mean: {permuted_mean:.4f}")
    print(f"\tPermuted Scores Standard Deviation: {permuted_std:.4f}")

    # Statistical significance check
    if p_value < 0.05:
        print(f"\tResult is statistically significant (p < 0.05).")
    else:
        print(f"\tResult is not statistically significant (p >= 0.05).")

def permutation_test(model, X_train, y_train, X_test, y_test, n_permutations=1000, random_state=None):
    """
    Performs a permutation test by training the model on the training set (with permuted labels)
    and evaluating it on the test set. Calculates the p-value based on the distribution of
    balanced accuracies from the permuted datasets.

    Parameters:
    - model: The machine learning model to evaluate.
    - X_train: Features of the training set.
    - y_train: Labels of the training set.
    - X_test: Features of the test set.
    - y_test: Labels of the test set.
    - n_permutations: Number of permutations to perform.
    - random_state: Seed for reproducibility.

    Returns:
    - p_value: The p-value from the permutation test.
    - observed_score: The observed balanced accuracy with true labels.
    - permuted_scores: Array of balanced accuracies from permuted labels.
    """
    rng = np.random.RandomState(random_state)

    # Step 1: Compute the observed performance using the true labels
    model_clone = clone(model)
    model_clone.fit(X_train, y_train)
    y_pred = model_clone.predict(X_test)
    observed_score = balanced_accuracy_score(y_test, y_pred)

    # Step 2: Permutation testing
    permuted_scores = []
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        # Shuffle the labels (y_train)
        y_permuted = shuffle(y_train, random_state=rng)

        # Clone and train the model on the permuted labels
        permuted_model = clone(model)
        permuted_model.fit(X_train, y_permuted)

        # Predict on the test set
        y_pred_perm = permuted_model.predict(X_test)

        # Calculate balanced accuracy and store it
        permuted_score = balanced_accuracy_score(y_test, y_pred_perm)
        permuted_scores.append(permuted_score)

    permuted_scores = np.array(permuted_scores)

    # Step 3: Calculate the p-value
    p_value = np.mean(permuted_scores >= observed_score)

    return p_value, observed_score, permuted_scores

def bootstrap_confidence_intervals(model, X_train, y_train, X_test, y_test, n_bootstraps=1000, ci=0.95, random_state=42):
    """
    Computes percentile-based bootstrap confidence intervals for both train and test balanced accuracy.
    Uses custom_stratified_metrics (i.e. stratified CV) to compute the training balanced accuracy.

    Returns a dictionary with:
      'train IC lower', 'train IC upper', 'test IC lower', 'test IC upper'
    """
    rng = np.random.RandomState(random_state)
    train_scores = []
    test_scores = []
    # Get the unique classes (based on y_train)
    classes = np.unique(y_train)

    for i in tqdm(range(n_bootstraps), desc="Stratified Bootstrapping"):
        # Stratified bootstrapping: resample within each class
        indices = []
        for cls in classes:
            cls_indices = list(y_train.index[y_train == cls])
            sample_indices = list(rng.choice(cls_indices, size=len(cls_indices), replace=True))
            indices.extend(sample_indices)
        indices = np.array(indices)
        # Use .loc so that we keep the original indices
        X_resampled = X_train.loc[indices]
        y_resampled = y_train.loc[indices]

        # Train a clone of the model on the resampled data
        model_clone = clone(model)
        model_clone.fit(X_resampled, y_resampled)

        # Evaluate training balanced accuracy using custom stratified CV on the resampled training set
        train_metrics = custom_stratified_metrics(clone(model), X_resampled, y_resampled, cv_splits=5)
        train_score = train_metrics['Balanced_Accuracy']

        # Evaluate test balanced accuracy (fixed test set)
        y_test_pred = model_clone.predict(X_test)
        test_score = balanced_accuracy_score(y_test, y_test_pred)

        train_scores.append(train_score)
        test_scores.append(test_score)

    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)

    lower_percentile = (100 - ci * 100) / 2
    upper_percentile = 100 - lower_percentile
    train_lower_bound = np.percentile(train_scores, lower_percentile)
    train_upper_bound = np.percentile(train_scores, upper_percentile)
    test_lower_bound = np.percentile(test_scores, lower_percentile)
    test_upper_bound = np.percentile(test_scores, upper_percentile)

    return {
        'train IC lower': train_lower_bound,
        'train IC upper': train_upper_bound,
        'test IC lower': test_lower_bound,
        'test IC upper': test_upper_bound,
    }

def remove_random_state(model):
    if hasattr(model, "random_state"):
        model.random_state = None
    return model

def repeated_cv_confidence_intervals(model, X_train, y_train, n_repeats=1000, n_folds=5, ci=0.95):
    """
    Perform repeated cross-validation to estimate confidence intervals.

    Returns:
    - Dictionary with lower and upper bounds of the confidence interval
    """
    cv_scores = []

    # Remove the random state from the model
    model = remove_random_state(model)

    # Perform multiple cross-validation runs
    for _ in tqdm(range(n_repeats), desc="Repeated Cross-Validation"):
        fold_scores = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model_clone = clone(model)
            model_clone.fit(X_cv_train, y_cv_train)
            y_pred = model_clone.predict(X_cv_val)

            score = balanced_accuracy_score(y_cv_val, y_pred)
            fold_scores.append(score)

        # Store the mean CV score of this iteration
        cv_scores.append(np.mean(fold_scores))

    # Confidence Interval Calculation
    lower_percentile = (100 - ci * 100) / 2
    upper_percentile = 100 - lower_percentile

    lower_bound = np.percentile(cv_scores, lower_percentile)
    upper_bound = np.percentile(cv_scores, upper_percentile)

    plot_boot_scores_with_ci(cv_scores, confidence_level=ci)

    return {
        "CV mean": np.mean(cv_scores),
        "CV median": np.median(cv_scores),
        "CV IC lower": lower_bound,
        "CV IC upper": upper_bound,
    }

def correct_permutation_test(model, X_train, y_train, n_folds=5, n_permutations=1000, random_state=42):
    """
    Performs a permutation test using inner cross-validation to avoid test set leakage.
    """
    rng = np.random.RandomState(random_state)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(0, 2**32))
    observed_scores = []

    # Step 1: Compute the observed performance using inner CV
    for train_idx, val_idx in skf.split(X_train, y_train):
        model_clone = clone(model)
        model_clone.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_pred = model_clone.predict(X_train.iloc[val_idx])
        observed_scores.append(balanced_accuracy_score(y_train.iloc[val_idx], y_pred))

    observed_score = np.mean(observed_scores)

    # Step 2: Permutation Testing
    permuted_scores = []
    for _ in tqdm(range(n_permutations), desc="Permutation Test"):
        y_permuted = shuffle(y_train, random_state=rng.randint(0, 2**32)).reset_index(drop=True)  # Shuffle labels

        permuted_cv_scores = []
        for train_idx, val_idx in skf.split(X_train, y_permuted):
            model_clone = clone(model)
            model_clone.fit(X_train.iloc[train_idx], y_permuted.iloc[train_idx])
            y_pred_perm = model_clone.predict(X_train.iloc[val_idx])
            permuted_cv_scores.append(balanced_accuracy_score(y_permuted.iloc[val_idx], y_pred_perm))

        permuted_scores.append(np.mean(permuted_cv_scores))

    permuted_scores = np.array(permuted_scores)

    # Step 3: Calculate p-value
    p_value = np.mean(permuted_scores >= observed_score)

    return p_value, observed_score, permuted_scores

def custom_stratified_metrics(model, X, y, cv_splits=5, random_state=None):
    """
    Evaluates a model using stratified k-fold CV and computes precision, recall, F1 per class,
    along with balanced accuracy and overall accuracy.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    classes = np.unique(y)
    precision = dict(zip(classes, precision_score(y, y_pred, average=None, zero_division=0)))
    recall = dict(zip(classes, recall_score(y, y_pred, average=None, zero_division=0)))
    f1 = dict(zip(classes, f1_score(y, y_pred, average=None, zero_division=0)))
    bal_acc = balanced_accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)

    return {'Precision': precision, 'Recall': recall, 'F1': f1,
            'Balanced_Accuracy': bal_acc, 'Accuracy': acc}

