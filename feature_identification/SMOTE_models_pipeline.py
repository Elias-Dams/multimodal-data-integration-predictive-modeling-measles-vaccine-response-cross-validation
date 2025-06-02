import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.base import clone
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from helper_functions.utils import (
    handle_missing_values,
    load_groups_from_json,
    compress_correlated_features,
    encode_labels,
    split_dataset, scale_features, bootstrap_confidence_intervals, permutation_test, custom_stratified_metrics,
    correct_permutation_test, repeated_cv_confidence_intervals, check_missing_values, custom_train_test_split
)
from helper_functions.load_datasets import get_measles_data, get_hepatitis_data

###########################################################################
# CONFIG
###########################################################################

VACCINE = "Measles"

COMPRESS_CORRELATED = False # compress correlated features
BALANCE = None # if "custom" Create custom class weights else balanced.
OVERSAMPLING_METHODS =  [None, 'smote', 'smote-borderline', 'smote-adasyn', 'smote-smotetomek','smote-smoteenn']
SAVE_FILE = "TIME" #
SAVE_DIR = f"../data_created/SMOTE/{VACCINE}"

RANDOM_STATE = 42

start_time = time.time()

###########################################################################
# LOAD ALL THE DATA
###########################################################################

if VACCINE == "Measles":
    datasets_merged, abtiters = get_measles_data()
elif VACCINE == "Hepatitis":
    datasets_merged, abtiters = get_hepatitis_data()

else:
    raise ValueError(f"VACCINE = {VACCINE} which is not an option")

###########################################################################
# HANDLE MISSING VALUES
###########################################################################

for dataset in datasets_merged:
    check_missing_values(datasets_merged[dataset]["df"],dataset, len(abtiters))

###########################################################################
# COMPRESS CORRELATED FEATURES
###########################################################################
if COMPRESS_CORRELATED:
    if VACCINE == "Measles":
        cytokines_groups = load_groups_from_json(f"../data/{VACCINE}/clusters/cytokines.json")
        cytometry_groups = load_groups_from_json(f"../data/{VACCINE}/clusters/cytometry.json")
        rna_groups = load_groups_from_json(f"../data/{VACCINE}/clusters/RNA1.json") #TODO changed

        datasets_merged['cytokines']["df"] = compress_correlated_features(datasets_merged['cytokines']["df"], cytokines_groups)
        datasets_merged['cytometry']["df"] = compress_correlated_features(datasets_merged['cytometry']["df"], cytometry_groups)
        datasets_merged['RNa_data']["df"] = compress_correlated_features(datasets_merged['RNa_data']["df"], rna_groups)
    elif VACCINE == "Hepatitis":
        cytometry_groups = load_groups_from_json(f"../data/{VACCINE}/clusters/cytometry.json")
        rna_groups = load_groups_from_json(f"../data/{VACCINE}/clusters/RNA1.json") #TODO changed

        datasets_merged['cytometry']["df"] = compress_correlated_features(datasets_merged['cytometry']["df"],cytometry_groups)
        datasets_merged['RNa_data']["df"] = compress_correlated_features(datasets_merged['RNa_data']["df"], rna_groups)
    else:
        raise ValueError(f"VACCINE = {VACCINE} which is not an option")

###########################################################################
# ENCODE THE LABELS (responder -> 1, non responder -> 0)
###########################################################################

for dataset in datasets_merged:
    datasets_merged[dataset]["df"] = encode_labels(datasets_merged[dataset]["df"])

###########################################################################
# START LOOP
###########################################################################

original_dfs = {dataset: df_dict["df"].copy(deep=True) for dataset, df_dict in datasets_merged.items()}

vaccinees = abtiters['Vaccinee']

# STRATIFIED SPLIT
# train_vaccinees, test_vaccinees = train_test_split(vaccinees, test_size=0.2, random_state=RANDOM_STATE,
#                                                    stratify=abtiters['response_label'])
# if VACCINE == "Measles":
#     train_vaccinees_exceptions, test_vaccinees_exceptions = train_test_split(datasets_merged['clonal_depth']["df"]["Vaccinee"], test_size=0.2, random_state=RANDOM_STATE,
#                                                        stratify=datasets_merged['clonal_depth']["df"]['response_label'])

# BALANCED SPLIT
train_vaccinees, test_vaccinees = custom_train_test_split(vaccinees, abtiters['response_label'], test_size=0.2, random_state=RANDOM_STATE)
if VACCINE == "Measles":
    train_vaccinees_exceptions, test_vaccinees_exceptions = custom_train_test_split(
        datasets_merged['clonal_depth']["df"]["Vaccinee"], datasets_merged['clonal_depth']["df"]['response_label'], test_size=0.2, random_state=RANDOM_STATE)

for oversample_method in OVERSAMPLING_METHODS:
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
        if dataset in ['clonal_depth', 'clonal_breadth']:
            X_train, X_test, y_train, y_test, _ = split_dataset(datasets_merged[dataset]["df"], train_vaccinees_exceptions, test_vaccinees_exceptions, oversampling_method=OVERSAMPLING_METHOD, random_state=RANDOM_STATE)
        else:
            X_train, X_test, y_train, y_test, _ = split_dataset(datasets_merged[dataset]["df"], train_vaccinees, test_vaccinees, oversampling_method=OVERSAMPLING_METHOD, random_state=RANDOM_STATE)
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
            'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight=custom_class_weights),
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,
                                                      class_weight=custom_class_weights),
            'SVM': SVC(random_state=RANDOM_STATE, class_weight=custom_class_weights, probability=True),
            'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=custom_class_weights),
            'Naive Bayes': GaussianNB()
        }

        # Evaluate each model using stratified k-fold CV
        evaluation_results = []
        for name, model in models.items():
            # metrics = custom_stratified_metrics(cloned_model, X_train.copy(), y_train.copy(), cv_splits=5,
            #                                     random_state=RANDOM_STATE)
            #
            # p_value, observed_score, _ = permutation_test(
            #     model, X_train_scaled, y_train, X_test_scaled, y_test,
            #     n_permutations=1000, random_state=RANDOM_STATE
            # )
            #
            # confidence_intervals = bootstrap_confidence_intervals(
            #     model, X_train_scaled, y_train, X_test_scaled, y_test,
            #     n_bootstraps=1000, ci=0.95, random_state=RANDOM_STATE
            # )

            metrics = repeated_cv_confidence_intervals(clone(model), X_train, y_train, n_repeats=1000, n_folds=5, ci=0.95)

            p_value, observed_score, _ = correct_permutation_test(clone(model), X_train, y_train, n_folds=5, n_permutations=1000, random_state=RANDOM_STATE)
            print(f"CV MEAN: {metrics['CV mean']}, CV MEDIAN: {metrics['CV median']}, OBSERVED SCORE: {observed_score}")

            # Train on full training data and calculate test accuracy
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_bal_acc = balanced_accuracy_score(y_test, y_pred)

            # Calculate and visualize confusion matrix
            report = classification_report(y_test, y_pred, output_dict=True)  # added output_dict to get per class metrics.

            result_entry = {
                'Model': name,
                **metrics,
                'p-value': p_value,
                'Test Accuracy': test_acc,
                'Test Balanced_Accuracy': test_bal_acc,
                'Test Classification Report': report,
                # The 'Train Vaccinees' and 'Test Vaccinees' keys will be added below
            }

            # Conditionally determine and add the correct vaccine lists based on the data_name
            if data_name in ['clonal_depth', 'clonal_breadth']:
                result_entry['Train Vaccinees'] = list(train_vaccinees_exceptions.items())
                result_entry['Test Vaccinees'] = list(test_vaccinees_exceptions.items())
            else:
                result_entry['Train Vaccinees'] = list(train_vaccinees.items())
                result_entry['Test Vaccinees'] = list(test_vaccinees.items())

            evaluation_results.append(result_entry)

        results_df = pd.DataFrame(evaluation_results)

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

    for dataset in datasets_merged:
        if datasets_merged[dataset]["split"]['X_train'] is not None and \
                datasets_merged[dataset]["split"]['X_test'] is not None and \
                datasets_merged[dataset]["split"]['y_train'] is not None and \
                datasets_merged[dataset]["split"]['y_test'] is not None:
            X_train_scaled = datasets_merged[dataset]["split"]['X_train']
            y_train = datasets_merged[dataset]["split"]['y_train']
            X_test_scaled = datasets_merged[dataset]["split"]['X_test']
            y_test = datasets_merged[dataset]["split"]['y_test']

            train_and_predict_best_model(X_train_scaled, y_train, X_test_scaled, y_test, dataset)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")