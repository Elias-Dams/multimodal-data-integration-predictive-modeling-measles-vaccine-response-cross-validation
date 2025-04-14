import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.base import clone
import numpy as np
import pandas as pd
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import compute_class_weight
import joblib  # A library for efficiently serializing Python objects


from exploration.SMOTE_utils import (
    handle_missing_values,
    load_groups_from_json,
    compress_correlated_features,
    encode_labels,
    split_dataset, scale_features, bootstrap_confidence_intervals, permutation_test, custom_stratified_metrics,
    correct_permutation_test, repeated_cv_confidence_intervals, train_val_test_split, oversample_data
)
from exploration.load_datasets import get_measles_data, get_hepatitis_data

###########################################################################
# CONFIG
###########################################################################

VACCINE = "Measles"

COMPRESS_CORRELATED = False # compress correlated features
OVERSAMPLING_METHODS = [None, 'random', 'smote', ] #'smote-borderline', 'smote-adasyn', 'smote-smotetomek','smote-smoteenn']
SAVE_FILE = "TEST_UNCOMPRESSED"
SAVE_DIR = f"../data_created/CONSENSUS/{VACCINE}"

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
    datasets_merged[dataset]["df"] = handle_missing_values(datasets_merged[dataset]["df"],dataset, abtiters, strategy='mean')

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

# to just shuffle randomly
vaccinees = abtiters['Vaccinee']
train_vaccinees, val_vaccinees, test_vaccinees = train_val_test_split(
    vaccinees, abtiters['response_label'], train_size=0.7, val_size=0.15, test_size=0.15, random_state=42, stratify=abtiters['response_label']
)

best_models_per_dataset = {}

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

        # Use the pre-defined train and test vaccinees
        curr_df = datasets_merged[dataset]["df"]

        train_df = curr_df[curr_df['Vaccinee'].isin(train_vaccinees)]
        val_df = curr_df[curr_df['Vaccinee'].isin(val_vaccinees)]
        test_df = curr_df[curr_df['Vaccinee'].isin(test_vaccinees)]

        # Separate features and labels
        X_train = train_df.drop(['Vaccinee', 'response_label'], axis=1)
        y_train = train_df['response_label']
        X_val = val_df.drop(['Vaccinee', 'response_label'], axis=1)
        y_val = val_df['response_label']
        X_test = test_df.drop(['Vaccinee', 'response_label'], axis=1)
        y_test = test_df['response_label']

        X_train_resampled = X_train
        y_train_resampled = y_train

        pca = None

        if OVERSAMPLING_METHOD is not None:
            X_train_resampled, y_train_resampled = oversample_data(X_train, y_train, OVERSAMPLING_METHOD)

        else:
            # No oversampling
            print(f"Class distribution for train set: {Counter(y_train_resampled)}")
            print(f"Class distribution for test set: {Counter(y_test)}")

        split = {
            "X_train": X_train_resampled,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train_resampled,
            "y_val": y_val,
            "y_test": y_test,
        }
        datasets_merged[dataset]["split"] = split

    ###########################################################################
    # SCALE THE FEATURES
    ###########################################################################

    for dataset in list(datasets_merged.keys()):
        if (datasets_merged[dataset]["split"]['X_train'] is not None
            and datasets_merged[dataset]["split"]['X_test'] is not None
            and datasets_merged[dataset]["split"]['X_val'] is not None):

            X_train = datasets_merged[dataset]["split"]['X_train']
            X_val = datasets_merged[dataset]["split"]['X_val']
            X_test = datasets_merged[dataset]["split"]['X_test']

            scaler = StandardScaler()
            # Fit and transform the training data, and transform the test data
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            # Wrap the numpy arrays back into DataFrames with the original indices and column names
            X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
            X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

            datasets_merged[dataset]["split"]['X_train'] = X_train_scaled
            datasets_merged[dataset]["split"]['X_val'] = X_val_scaled
            datasets_merged[dataset]["split"]['X_test'] = X_test_scaled

    ###########################################################################
    # TRAIN AND PREDICT BEST MODEL
    ###########################################################################

    def train_and_predict_best_model(X_train, y_train, X_val, y_val, X_test, y_test, data_name):
        """
        Trains multiple baseline models using stratified k-fold CV (minimal tuning) to evaluate their performance.
        The best model is selected based on a composite score from F1 and balanced accuracy.
        The function then trains the best model on the full training data and predicts the test set.
        Evaluation results are saved to a CSV file.
        """
        import warnings
        warnings.filterwarnings('ignore')

        # Calculate class weights
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

        best_model = None
        best_composite_score = -1
        validation_results = []

        # Evaluate each model on the validation set
        tolerance = 0.01  # Define a small tolerance for tie-breaking
        for name, model in models.items():
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            metrics = repeated_cv_confidence_intervals(model_clone, X_train, y_train, n_repeats=1000, n_folds=5,
                                                       ci=0.95)
            p_value, observed_score, _ = correct_permutation_test(model_clone, X_train, y_train, n_folds=5,
                                                                  n_permutations=1000)

            y_val_pred = model_clone.predict(X_val)

            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            val_bal_acc = balanced_accuracy_score(y_val, y_val_pred)
            current_composite_score = val_f1 + val_bal_acc

            validation_results.append({
                'Model': name,
                'Validation F1': val_f1,
                'Validation Balanced Accuracy': val_bal_acc,
                'Validation Composite Score': current_composite_score,
                'CV Mean (Val Metric)': metrics.get('CV mean', None),  # Assuming CV was done for a relevant metric
                'P-value': p_value
            })

            if current_composite_score > best_composite_score:
                best_composite_score = current_composite_score
                best_model = clone(model)
                best_model_name = name
                best_model_training_metrics = metrics
                best_model_p_value = p_value
            elif abs(current_composite_score - best_composite_score) <= tolerance:
                # Tie-breaker: Higher CV mean (assuming it's a relevant metric)
                if metrics.get('CV mean', -1) > best_model_training_metrics.get('CV mean', -2):
                    best_composite_score = current_composite_score
                    best_model = clone(model)
                    best_model_name = name
                    best_model_training_metrics = metrics
                    best_model_p_value = p_value
                # Further tie-breaker: Lower p-value (more significant)
                elif abs(metrics.get('CV mean', -1) - best_model_training_metrics.get('CV mean',
                                                                                      -2)) <= 0.001 and p_value < best_model_p_value:
                    best_composite_score = current_composite_score
                    best_model = clone(model)
                    best_model_name = name
                    best_model_training_metrics = metrics
                    best_model_p_value = p_value

        print(
            f"\n{data_name}: Best model based on validation set: {best_model_name} (oversampling: {OVERSAMPLING_METHOD}) (Validation Composite Score: {best_composite_score:.4f})")
        validation_results_df = pd.DataFrame(validation_results)
        print("\nValidation Results:\n", validation_results_df)

        # Train the best model on the combined training and validation data
        X_train_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_train_combined = pd.concat([y_train, y_val], ignore_index=True)
        best_model.fit(X_train_combined, y_train_combined)

        # Predict on the test set
        y_pred_test = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test, output_dict=True)

        # Prepare evaluation results for saving
        evaluation_results = [{
            'Model': best_model_name,
            'Train CV Mean': validation_results_df[validation_results_df['Model'] == best_model_name]['CV Mean (Val Metric)'].iloc[0],
            'Train P-value': validation_results_df[validation_results_df['Model'] == best_model_name]['P-value'].iloc[0],
            'Validation F1': validation_results_df[validation_results_df['Model'] == best_model_name]['Validation F1'].iloc[0],
            'Validation Balanced Accuracy': validation_results_df[validation_results_df['Model'] == best_model_name]['Validation Balanced Accuracy'].iloc[0],
            'Validation Composite Score': best_composite_score,
            'Test Accuracy': test_acc,
            'Test Balanced_Accuracy': test_bal_acc,
            'Test Classification Report': report,
            'Train Vaccinees': list(train_vaccinees.items()),
            'Validate Vaccinees': list(val_vaccinees.items()),
            'Test Vaccinees': list(test_vaccinees.items()),
            'Compress_Correlated': COMPRESS_CORRELATED,
            'Oversampling_Method': OVERSAMPLING_METHOD,
        }]

        results_df = pd.DataFrame(evaluation_results)

        # Calculate class support counts based on y_train_combined
        unique_classes, counts = np.unique(y_train_combined, return_counts=True)
        support_counts = dict(zip(unique_classes, counts))
        print("Support counts (combined train + val):", support_counts)
        counts_json = {}
        for cls, count in support_counts.items():
            counts_json[str(cls)] = int(count)
        results_df['Support_classed'] = json.dumps(counts_json)

        # Reorder the DataFrame
        cols = list(results_df.columns)
        cols_without = [col for col in cols if col not in ['Train Vaccinees', 'Test Vaccinees']]
        new_order = cols_without + ['Train Vaccinees', 'Test Vaccinees']
        results_df = results_df[new_order]

        # Save evaluation result
        results_file = os.path.join(SAVE_DIR, f"{SAVE_FILE}_{data_name}_data.csv")
        if os.path.exists(results_file):
            existing_df = pd.read_csv(results_file)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_csv(results_file, index=False)
        else:
            results_df.to_csv(results_file, index=False)

        best_model_filename = os.path.join(
            "../data_created/CONSENSUS/best_models",
            f"{SAVE_FILE}_{data_name}_{best_model_name}_{OVERSAMPLING_METHOD}_model.joblib"
        )
        try:
            joblib.dump(best_model, best_model_filename)
            print(f"Best model '{best_model_name}' saved to: {best_model_filename}")
        except Exception as e:
            print(f"Error saving the best model: {e}")

        # Construct the filename for the test set CSV
        test_set_csv_filename = os.path.join(
            "../data_created/CONSENSUS/test_sets",
            f"{SAVE_FILE}_{data_name}_test_set.csv"
        )

        # Save the test set to CSV using pandas
        try:
            X_test_df = pd.DataFrame(X_test)  # Convert X_test to DataFrame
            y_test_series = pd.Series(y_test, name='target')  # Convert y_test to Series with a name
            test_set_df = pd.concat([X_test_df, y_test_series], axis=1)
            test_set_df.to_csv(test_set_csv_filename, index=False)
            print(f"Test set saved to CSV: {test_set_csv_filename}")
        except Exception as e:
            print(f"Error saving the test set to CSV: {e}")

    for dataset in datasets_merged:
        if datasets_merged[dataset]["split"]['X_train'] is not None and \
                datasets_merged[dataset]["split"]['X_val'] is not None and \
                datasets_merged[dataset]["split"]['X_test'] is not None and \
                datasets_merged[dataset]["split"]['y_train'] is not None and \
                datasets_merged[dataset]["split"]['y_val'] is not None and \
                datasets_merged[dataset]["split"]['y_test'] is not None:

            X_train_scaled = datasets_merged[dataset]["split"]['X_train']
            y_train = datasets_merged[dataset]["split"]['y_train']
            X_val_scaled = datasets_merged[dataset]["split"]['X_val']
            y_val = datasets_merged[dataset]["split"]['y_val']
            X_test_scaled = datasets_merged[dataset]["split"]['X_test']
            y_test = datasets_merged[dataset]["split"]['y_test']

            train_and_predict_best_model(X_train_scaled, y_train,
                                         X_val_scaled, y_val,
                                         X_test_scaled, y_test,
                                         dataset)
