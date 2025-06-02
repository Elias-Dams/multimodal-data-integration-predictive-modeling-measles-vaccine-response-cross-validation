import json
import random
import time

from sklearn.model_selection import train_test_split, StratifiedKFold
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
import matplotlib.pyplot as plt
import seaborn as sns

from helper_functions.utils import (
    handle_missing_values,
    load_groups_from_json,
    compress_correlated_features,
    encode_labels,
    split_dataset, scale_features,
    check_missing_values,
    custom_train_test_split
)
from helper_functions.load_datasets import get_measles_data, get_hepatitis_data

start_time = time.time()

###########################################################################
# CONFIG
###########################################################################

VACCINE = "Hepatitis"

COMPRESS_CORRELATED = False # compress correlated features
OVERSAMPLING_METHODS =  [None, 'smote', 'smote-borderline', 'smote-adasyn', 'smote-smotetomek'] # 'smote-smoteenn'
SAVE_FILE = "TEST_UNCOMPRESSED_BALANCED_GENERAL_5000_SPLITS" #
SAVE_DIR = f"../data_created/SMOTE/{VACCINE}"

RANDOM_STATE = 42
RANDOM_STATE_SPLIT = random.sample(range(0, 10000), 5000)

###########################################################################
# LOAD ALL THE DATA
###########################################################################

if VACCINE == "Measles":
    datasets_merged, abtiters = get_measles_data()
elif VACCINE == "Hepatitis":
    datasets_merged, abtiters = get_hepatitis_data()
    del datasets_merged["RNa_data"]
    del datasets_merged["TCR_predictions"]
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
        rna_groups = load_groups_from_json(f"../data/{VACCINE}/clusters/RNA1.json")

        datasets_merged['cytokines']["df"] = compress_correlated_features(datasets_merged['cytokines']["df"], cytokines_groups)
        datasets_merged['cytometry']["df"] = compress_correlated_features(datasets_merged['cytometry']["df"], cytometry_groups)
        datasets_merged['RNa_data']["df"] = compress_correlated_features(datasets_merged['RNa_data']["df"], rna_groups)
    elif VACCINE == "Hepatitis":
        cytometry_groups = load_groups_from_json(f"../data/{VACCINE}/clusters/cytometry.json")
        rna_groups = load_groups_from_json(f"../data/{VACCINE}/clusters/RNA1.json")

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

for random_state_split in RANDOM_STATE_SPLIT:

    # BALANCED SPLIT
    train_vaccinees, test_vaccinees = custom_train_test_split(vaccinees, abtiters['response_label'], test_size=0.2, random_state=random_state_split)
    if VACCINE == "Measles":
        train_vaccinees_exceptions, test_vaccinees_exceptions = custom_train_test_split(
            datasets_merged['clonal_depth']["df"]["Vaccinee"], datasets_merged['clonal_depth']["df"]['response_label'], test_size=0.2, random_state=random_state_split)

    for oversample_method in OVERSAMPLING_METHODS:
        print(f"Oversampling {oversample_method}")
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
            balanced_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            custom_class_weights = dict(zip(np.unique(y_train), balanced_weights))

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

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

                cv_fold_bal_acc_scores = []

                for fold_idx, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):

                    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

                    model_fold = clone(model)
                    model_fold.fit(X_train_fold, y_train_fold)
                    y_val_pred = model_fold.predict(X_val_fold)
                    fold_bal_acc = balanced_accuracy_score(y_val_fold, y_val_pred)
                    cv_fold_bal_acc_scores.append(fold_bal_acc)

                mean_cv_bal_acc = np.mean(cv_fold_bal_acc_scores)
                median_cv_bal_acc = np.median(cv_fold_bal_acc_scores)

                # Train on full training data and calculate test accuracy
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                test_bal_acc = balanced_accuracy_score(y_test, y_pred)

                # Calculate and visualize confusion matrix
                report = classification_report(y_test, y_pred, output_dict=True)

                result_entry = {
                    'Model': name,
                    'CV mean': mean_cv_bal_acc,
                    'CV median': median_cv_bal_acc,
                    'Test Accuracy': test_acc,
                    'Test Balanced_Accuracy': test_bal_acc,
                    'Test Classification Report': report,
                    'Split seed': random_state_split
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
                results_df.to_csv(results_file, mode='a', header=False, index=False)
            else:
                results_df.to_csv(results_file, mode='w', index=False)

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