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

from exploration.SMOTE_utils import (
    handle_missing_values,
    load_groups_from_json,
    compress_correlated_features,
    encode_labels,
    split_dataset, scale_features, bootstrap_confidence_intervals, permutation_test, custom_stratified_metrics
)

###########################################################################
# CONFIG
###########################################################################

LABELS = {'responder': {'name': 'response', 'color': 'blue'},
          'no response - high ab': {'name': 'no response', 'color': 'orange'},
          'no response - low ab':  {'name': 'no response', 'color': 'green'}
          }
COMPRESS_CORRELATED = False # compress correlated features
OVERSAMPLING_METHOD = 'smote' # oversample the minority class ['remove', 'random', 'smote', 'smote-borderline', 'smote-adasyn', 'smote-smotetomek', 'smote-smoteenn'] if None use balance accuracy
BALANCE = None # if "custom" Create custom class weights else balanced.
SAVE_FILE = "TEST_UNCOMPRESSED" #
SAVE_DIR = "../data_created/SMOTE/Measles"

###########################################################################
# LOAD ALL THE DATA
###########################################################################

# Load the antibody titers data (response profile)
abtiters = pd.read_csv('../data/Measles/antibody_df.csv')
print("Antibody titers data loaded. Shape:", abtiters.shape)
print(abtiters.head())

# Load the cytokines data
cytokines = pd.read_csv('../data/Measles/cytokines_data.csv')
print("Cytokines data loaded. Shape:", cytokines.shape)
print(cytokines.head())

# Load the cytometry data
cytometry = pd.read_csv('../data/Measles/cyto_data.csv')
print("Cytometry data loaded. Shape:", cytometry.shape)
print(cytometry.head())

# Load clonal breadth data
clonal_breadth = pd.read_csv('../data/Measles/clonal_breadth_data.csv')
print("Clonal breadth data loaded. Shape:", clonal_breadth.shape)
print(clonal_breadth.head())
# How many different TCR clonotypes (unique combinations of sequences, V/J genes, etc.) are specific to measles.

# Load clonal depth data
clonal_depth = pd.read_csv('../data/Measles/clonal_depth_data.csv')
print("Clonal depth data loaded. Shape:", clonal_depth.shape)
print(clonal_depth.head())
# The ratio of unique beta-chain sequences predicted to bind measles to the total number of clonotypes.
# This focuses on the most important part of the TCR (the beta chain).

# Load the module scores
module_scores = pd.read_csv('../data/Measles/RNA_circos.csv')
print("Module scores loaded. Shape:", module_scores.shape)
print(module_scores.head())

# Check the distribution of module scores
print("Module score columns:", module_scores.columns[1:10])

datasets = {
    "antibody_titers": abtiters,
    "cytokines": cytokines,
    "cytometry": cytometry,
    "clonal_breadth": clonal_breadth,
    "clonal_depth": clonal_depth,
    "RNa_data": module_scores,
}

###########################################################################
# ASSIGN LABELS
###########################################################################

# Count the frequency of each label
abtiters['response_label'] = abtiters['quadrant'].replace({key: value['name'] for key, value in LABELS.items()})

# Count the frequency of each label
response_counts = abtiters['response_label'].value_counts()

print("\nFrequency of responses:")
print(response_counts)

converted_dict = {value['name']: value['color'] for value in LABELS.values()}

colors = [converted_dict[label] for label in response_counts.index]

# Plot the frequency of high vs. low responses as a bar plot
plt.figure(figsize=(8, 6))
response_counts.plot(kind='bar', color=colors)
plt.title('Frequency of High vs Low Titer Responses')
plt.xlabel('Response Label')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
for label, group in abtiters.groupby('response_label'):
    plt.scatter(group['Day 0'], group['Day 21'], color=converted_dict[label], label=label)

# Plot reference line y = x
plt.plot([0, max(abtiters['Day 0'])], [0, max(abtiters['Day 21'])], color='red', linestyle='-', linewidth=1)

# Labeling and legend
plt.xlabel('Day 0')
plt.ylabel('Day 21')
plt.legend(title="Titer Response")
plt.show()

abtiters = abtiters.drop(columns=['vaccine', 'Day 0', 'Day 21', 'Day 150', 'Day 365', 'diff: 21-0', 'diff: 150-21', 'diff: 365-150', 'response', 'protected','quadrant'])
## cytokines already loaded ##
cytometry = cytometry[['Vaccinee', 'WBC Day 0','RBC Day 0','HGB Day 0','HCT Day 0','PLT Day 0','%LYM Day 0','%MON Day 0','%GRA Day 0']]

datasets['antibody_titers'] = abtiters
datasets['cytometry'] = cytometry

###########################################################################
# MERGE DATASETS
###########################################################################

# Merge datasets with the labels
cytokines_merged = pd.merge(datasets['antibody_titers'], datasets['cytokines'], on='Vaccinee')
cytometry_merged = pd.merge(datasets['antibody_titers'], datasets['cytometry'], on='Vaccinee')
clonal_breadth_merged = pd.merge(datasets['antibody_titers'], datasets['clonal_breadth'], on='Vaccinee') #for now because they only contain 27 values
clonal_depth_merged = pd.merge(datasets['antibody_titers'], datasets['clonal_depth'], on='Vaccinee') #for now because they only contain 27 values
rna_merged = pd.merge(datasets['antibody_titers'], datasets['RNa_data'], on='Vaccinee')

datasets_merged = {
    "cytokines": {"df": cytokines_merged, "split": None},
    "cytometry": {"df": cytometry_merged, "split": None},
    "clonal_breadth": {"df": clonal_breadth_merged, "split": None},
    "clonal_depth": {"df": clonal_depth_merged, "split": None},
    "RNa_data": {"df": rna_merged, "split": None},
}

###########################################################################
# HANDLE MISSING VALUES
###########################################################################

for dataset in datasets_merged:
    datasets_merged[dataset]["df"] = handle_missing_values(datasets_merged[dataset]["df"],
                                                           dataset, datasets['antibody_titers'],
                                                           strategy='mean')

###########################################################################
# COMPRESS CORRELATED FEATURES
###########################################################################

if COMPRESS_CORRELATED:
    # The functions (compress_correlated_features and load_groups_from_json) are assumed to exist
    cytokines_groups = load_groups_from_json("../data/Measles/clusters/cytokines.json")
    cytometry_groups = load_groups_from_json("../data/Measles/clusters/cytometry.json")
    rna_groups = load_groups_from_json("../data/Measles/clusters/RNA1.json")

    datasets_merged['cytokines']["df"] = compress_correlated_features(datasets_merged['cytokines']["df"], cytokines_groups)
    datasets_merged['cytometry']["df"] = compress_correlated_features(datasets_merged['cytometry']["df"], cytometry_groups)
    datasets_merged['RNa_data']["df"] = compress_correlated_features(datasets_merged['RNa_data']["df"], rna_groups)

###########################################################################
# ENCODE THE LABELS (responder -> 1, non responder -> 0)
###########################################################################

for dataset in datasets_merged:
    datasets_merged[dataset]["df"] = encode_labels(datasets_merged[dataset]["df"])

###########################################################################
# START LOOP
###########################################################################

original_dfs = {dataset: df_dict["df"].copy(deep=True) for dataset, df_dict in datasets_merged.items()}

# to just shuffle randomly
vaccinees = abtiters['Vaccinee']
train_vaccinees, test_vaccinees = train_test_split(vaccinees, test_size=0.2, random_state=42,stratify=abtiters['response_label'])

oversampling_methods = ['smote-smoteenn'] # None, 'smote', 'smote-borderline', 'smote-adasyn', 'smote-smotetomek',
for oversample_method in oversampling_methods:
    OVERSAMPLING_METHOD = oversample_method

    # Sanity check: Verify that each dataset's "df" remains unchanged
    for dataset in datasets_merged:
        if not datasets_merged[dataset]["df"].equals(original_dfs[dataset]):
            raise ValueError(
                f"DataFrame for {dataset} has been modified in oversampling iteration {oversample_method}!")
        else:
            print(f"DataFrame for {dataset} remains unchanged for oversampling iteration {oversample_method}.")


    ###########################################################################
    # SPLIT THE DATASET
    ###########################################################################

    for dataset in datasets_merged:
        X_train, X_test, y_train, y_test = split_dataset(datasets_merged[dataset]["df"], train_vaccinees, test_vaccinees, oversampling_method=OVERSAMPLING_METHOD)
        split = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        datasets_merged[dataset]["split"] = split

    ###########################################################################
    # SCALE THE FEATURES
    ###########################################################################

    for dataset in list(datasets_merged.keys()):
        if datasets_merged[dataset]["split"]['X_train'] is not None and datasets_merged[dataset]["split"]['X_test'] is not None:
            X_train_scaled, X_test_scaled = scale_features(datasets_merged[dataset]["split"]['X_train'],datasets_merged[dataset]["split"]['X_test'])
            datasets_merged[dataset]["split"]['X_train'] = X_train_scaled
            datasets_merged[dataset]["split"]['X_test'] = X_test_scaled


    ###########################################################################
    # TRAIN AND PREDICT BEST MODEL
    ###########################################################################

    def train_and_predict_best_model(X_train, y_train, X_test, y_test, data_name):
        """
        Trains multiple baseline models using stratified k-fold CV (minimal tuning) to evaluate their performance.
        The best model is selected based on a composite score from F1 and balanced accuracy.
        The function then trains the best model on the full training data and predicts the test set.
        Evaluation results are saved to a CSV file.
        """
        import warnings
        warnings.filterwarnings('ignore')

        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        if BALANCE == 'custom':
            unique, counts = np.unique(y_train, return_counts=True)
            max_count = counts.max()
            custom_class_weights = {label: (1 / count) * max_count for label, count in zip(unique, counts)}
        else:
            balanced_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            custom_class_weights = dict(zip(np.unique(y_train), balanced_weights))
        print("Class Weights:", custom_class_weights)

        # Define models with balanced class weights where applicable
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, class_weight=custom_class_weights),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000,
                                                      class_weight=custom_class_weights),
            'SVM': SVC(random_state=42, class_weight=custom_class_weights, probability=True),
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight=custom_class_weights),
            'Naive Bayes': GaussianNB()
        }

        # Evaluate each model using stratified k-fold CV
        evaluation_results = []
        for name, model in models.items():
            cloned_model = clone(model)
            metrics = custom_stratified_metrics(cloned_model, X_train.copy(), y_train.copy(), cv_splits=5,
                                                random_state=42)

            p_value, observed_score, _ = permutation_test(
                model, X_train_scaled, y_train, X_test_scaled, y_test,
                n_permutations=1000, random_state=42
            )

            confidence_intervals = bootstrap_confidence_intervals(
                model, X_train_scaled, y_train, X_test_scaled, y_test,
                n_bootstraps=1000, ci=0.95, random_state=42
            )

            # Train on full training data and calculate test accuracy
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_bal_acc = balanced_accuracy_score(y_test, y_pred)

            # Combine all metrics
            evaluation_results.append({
                'Model': name,
                **metrics,
                'p-value': p_value,
                **confidence_intervals,
                'Test Accuracy': test_acc,
                'Test Balanced_Accuracy': test_bal_acc,
                'Train Vaccinees': list(train_vaccinees.items()),
                'Test Vaccinees': list(test_vaccinees.items())
            })

        results_df = pd.DataFrame(evaluation_results)

        # Function to flatten per-class metrics into separate columns
        def flatten_metrics(df, metric_name, classes):
            for cls in classes:
                df[f'{metric_name}_{cls}'] = df[metric_name].apply(lambda x: x.get(cls, 0))
            df.drop(columns=[metric_name], inplace=True)
            return df

        classes = np.unique(y_train)
        results_df = flatten_metrics(results_df, 'Precision', classes)
        results_df = flatten_metrics(results_df, 'Recall', classes)
        results_df = flatten_metrics(results_df, 'F1', classes)
        results_df['Compress_Correlated'] = COMPRESS_CORRELATED
        results_df['Oversampling_Method'] = OVERSAMPLING_METHOD
        # Calculate class support counts based on y_train
        unique_classes, counts = np.unique(y_train, return_counts=True)
        support_counts = dict(zip(unique_classes, counts))

        # Print the support counts dictionary
        print("Support counts:", support_counts)
        counts = {}
        for cls, count in support_counts.items():
            counts[str(cls)] = int(count)
        results_df['Support_classed'] = json.dumps(counts)

        # Reorder the DataFrame
        cols = list(results_df.columns)
        cols_without = [col for col in cols if col not in ['Train Vaccinees', 'Test Vaccinees']]
        new_order = cols_without + ['Train Vaccinees', 'Test Vaccinees']
        results_df = results_df[new_order]

        # Save evaluation results
        results_file = os.path.join(SAVE_DIR, f"{SAVE_FILE}_{data_name}_data.csv")
        if os.path.exists(results_file):
            existing_df = pd.read_csv(results_file)
            # Remove rows that have the same parameter values
            mask = (existing_df['Compress_Correlated'] == COMPRESS_CORRELATED) & \
                   (existing_df['Oversampling_Method'] == OVERSAMPLING_METHOD)
            existing_df = existing_df[~mask]
            # Combine the remaining rows with the new results
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_csv(results_file, index=False)
        else:
            results_df.to_csv(results_file, index=False)

        # Define metric weights for composite scoring (focus on F1 and balanced accuracy)
        metric_weights = {
            'Precision': 0,
            'Recall': 0,
            'F1': 2,  # Weight for F1 per class
            'Balanced_Accuracy': 4,  # Higher weight for Balanced Accuracy
            'Accuracy': 0
        }

        # Generate weights for per-class and scalar metrics
        def generate_dynamic_weights(metric_weights, classes):
            weights = {}
            for cls in classes:
                for metric in ['Precision', 'Recall', 'F1']:
                    weight = metric_weights.get(metric, 0)
                    weights[f"{metric}_{cls}"] = weight
            for scalar_metric in ['Balanced_Accuracy', 'Accuracy']:
                weights[scalar_metric] = metric_weights.get(scalar_metric, 0)
            return weights

        weights = generate_dynamic_weights(metric_weights, classes)

        # Check that all required columns exist
        missing_weights = set(weights.keys()) - set(results_df.columns)
        if missing_weights:
            raise ValueError(f"Missing weight keys in results DataFrame: {missing_weights}")

        # Calculate composite score
        metric_columns = list(weights.keys())
        results_df['Composite_Score'] = results_df[metric_columns].multiply(pd.Series(weights), axis=1).sum(axis=1)

        # Select the best model based on the composite score
        best_model_row = results_df.loc[results_df['Composite_Score'].idxmax()]
        best_model_name = best_model_row['Model']
        best_model_metrics = best_model_row.to_dict()

        # Train best model on full training data and predict on test set
        best_model = clone(models[best_model_name])
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Print weighted metrics for the best model
        filtered_metrics = {key: value for key, value in best_model_metrics.items()
                            if key not in ['Model', 'Composite_Score'] and weights.get(key, 0) > 0}
        best_model_df = pd.DataFrame([filtered_metrics])
        best_model_df.insert(0, 'Model', best_model_name)
        print("\nBest Model and Its Metrics (Weighted metrics only):")
        print(best_model_df)

        return y_pred, best_model_name, best_model, best_model_metrics

    for dataset in datasets_merged:
        if datasets_merged[dataset]["split"]['X_train'] is not None and \
                datasets_merged[dataset]["split"]['X_test'] is not None and \
                datasets_merged[dataset]["split"]['y_train'] is not None and \
                datasets_merged[dataset]["split"]['y_test'] is not None:
            X_train_scaled = datasets_merged[dataset]["split"]['X_train']
            y_train = datasets_merged[dataset]["split"]['y_train']
            X_test_scaled = datasets_merged[dataset]["split"]['X_test']
            y_test = datasets_merged[dataset]["split"]['y_test']

            y_pred, best_model_name, best_model, train_metrics = train_and_predict_best_model(X_train_scaled, y_train,
                                                                                              X_test_scaled, y_test, dataset)
            datasets_merged[dataset]["split"]['y_pred'] = y_pred
            datasets_merged[dataset]["split"]['best_model_name'] = best_model_name
            datasets_merged[dataset]["split"]['best_model'] = best_model
            datasets_merged[dataset]["split"]['train_metrics'] = train_metrics


