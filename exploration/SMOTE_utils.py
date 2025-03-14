from sklearn.preprocessing import StandardScaler
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from sklearn.utils import resample, shuffle
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from scipy.stats import bootstrap
import numpy as np
import pandas as pd
import os

from tqdm import tqdm


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

    def compress_features(data, features, new_feature_name):
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
    all_features_to_drop = [feature for group in groups.values() for feature in group]
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

def split_dataset(df, train_vaccinees, test_vaccinees, oversampling_method=None):
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

    if oversampling_method is not None:
        if oversampling_method == 'smote':
            # Perform oversampling on the training data
            sm = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
            # Creates new synthetic examples by interpolating between a minority class instance and one of its k-nearest neighbors.
            # This helps to enlarge the minority class without simply duplicating existing samples.

        elif oversampling_method == 'smote-borderline':
            # Perform oversampling on the training data
            sm = BorderlineSMOTE(kind='borderline-1', random_state=42)
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
            # Focuses on instances that lie near the decision boundary (the “borderline” samples).
            # By generating new examples only for those hard-to-learn cases, it aims to improve the classifier’s ability to separate classes where they overlap.

        elif oversampling_method == 'smote-adasyn':
            # Perform oversampling on the training data
            sm = ADASYN(random_state=42)
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
            # Similar to SMOTE, but it adaptively generates more synthetic examples for minority class samples that are harder to learn
            # (i.e., those with fewer minority neighbors). This way, the algorithm shifts the decision boundary toward the
            # majority class and focuses on areas with higher complexity.

        elif oversampling_method == 'smote-smotetomek':
            # Perform oversampling on the training data
            sm = SMOTETomek(random_state=42)
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
            # Combines SMOTE with an under-sampling technique called Edited Nearest Neighbors (ENN).
            # After oversampling with SMOTE, ENN removes samples that are misclassified by their neighbors.
            # This helps clean up noise and can improve model performance.

        elif oversampling_method == 'smote-smoteenn':
            # Perform oversampling on the training data
            sm = SMOTEENN(random_state=42)
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
            # Merges SMOTE with Tomek links under-sampling. Tomek links are pairs of samples from opposite classes that are each other’s nearest neighbors.
            # Removing these links after SMOTE oversampling can help eliminate overlapping samples and create a cleaner decision boundary.

        elif oversampling_method == 'random':
            # Perform random oversampling on the training data (simply duplicates existing minority class samples)
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

        elif oversampling_method == 'remove':
            # Perform random undersampling on the majority class
            majority_class = Counter(y_train).most_common(1)[0][0]  # Get the majority class label
            minority_class = [cls for cls in set(y_train) if cls != majority_class][0]  # Get minority class label

            # Separate majority and minority classes
            X_majority = X_train[y_train == majority_class]
            y_majority = y_train[y_train == majority_class]
            X_minority = X_train[y_train == minority_class]
            y_minority = y_train[y_train == minority_class]

            # Downsample majority class to match minority class size
            X_majority_downsampled, y_majority_downsampled = resample(
                X_majority, y_majority,
                replace=False,
                n_samples=len(y_minority),
                random_state=42
            )

            # Identify the removed majority class samples
            # Use index to ensure accurate removal
            removed_indices = X_majority.index.difference(X_majority_downsampled.index)
            X_majority_removed = X_majority.loc[removed_indices]
            y_majority_removed = y_majority.loc[removed_indices]

            # Combine downsampled majority class with the minority class
            X_train_resampled = pd.concat([X_majority_downsampled, X_minority])
            y_train_resampled = pd.concat([y_majority_downsampled, y_minority])

            # Add the removed samples to the rest set
            # Concatenate features and labels for the rest set
            removed_samples = pd.concat([X_majority_removed, y_majority_removed], axis=1)
            X_rest = removed_samples.drop(['response_label'], axis=1)
            y_rest = removed_samples['response_label']
            X_test = pd.concat([X_test, X_rest])
            y_test = pd.concat([y_test, y_rest])

        else:
            raise ValueError(f"Unsupported oversampling method: {oversampling_method}")

        # After oversampling
        print(f"Resampled class distribution using {oversampling_method} for train set: {Counter(y_train_resampled)}")
        print(f"Resampled class distribution using {oversampling_method} for test set: {Counter(y_test)}")

        # Check if oversampling left only one class
        if len(np.unique(y_train_resampled)) < 2:
            print(f"Only one class present after oversampling with {oversampling_method}. Skipping this dataset.")
            return None, None, None, None

    else:
        # No oversampling
        print(f"Class distribution for train set: {Counter(y_train_resampled)}")
        print(f"Class distribution for test set: {Counter(y_test)}")

    return X_train_resampled, X_test, y_train_resampled, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    # Fit and transform the training data, and transform the test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Wrap the numpy arrays back into DataFrames with the original indices and column names
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
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

def bootstrap_confidence_intervals(model, X_train, y_train, X_test, y_test, n_bootstraps=1000, ci=0.95,
                                   random_state=42):
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

