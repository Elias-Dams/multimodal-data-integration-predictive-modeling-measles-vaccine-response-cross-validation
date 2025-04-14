import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import numpy as np
import pandas as pd
import os

from exploration.SMOTE_utils import (
    handle_missing_values,
    load_groups_from_json,
    compress_correlated_features,
    encode_labels,
    split_dataset, scale_features, bootstrap_confidence_intervals, permutation_test, custom_stratified_metrics,
    repeated_cv_confidence_intervals
)
from exploration.load_datasets import get_measles_data, get_hepatitis_data

###########################################################################
# CONFIG
###########################################################################

VACCINE = "Measles"

COMPRESS_CORRELATED = False # compress correlated features
BALANCE = None # if "custom" Create custom class weights else balanced.
# OVERSAMPLING_METHODS = [None, 'smote', 'smote-borderline', 'smote-adasyn', 'smote-smotetomek','smote-smoteenn']
OVERSAMPLING_METHOD = 'smote'
SAVE_FILE = "TEST_HYPERPARAMETERS" #
SAVE_DIR = f"../data_created/SMOTE/{VACCINE}"

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
train_vaccinees, test_vaccinees = train_test_split(vaccinees, test_size=0.2, random_state=42,stratify=abtiters['response_label'])

###########################################################################
# SPLIT THE DATASET
###########################################################################

for dataset in datasets_merged:
    X_train, X_test, y_train, y_test, _ = split_dataset(datasets_merged[dataset]["df"], train_vaccinees, test_vaccinees, oversampling_method=OVERSAMPLING_METHOD)
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

    param_grid = {
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 4]
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [100, 200, 500]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        },
        "Decision Tree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Naive Bayes": {
            # GaussianNB has few hyperparameters, but you might consider:
            "var_smoothing": [1e-9, 1e-8, 1e-7]
        }
    }

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
        params = param_grid.get(name, None)
        grid_search = GridSearchCV(clone(model), params, cv=5,
                                   scoring='balanced_accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

        p_value, observed_score, _ = permutation_test(
            best_model, X_train_scaled, y_train, X_test_scaled, y_test,
            n_permutations=1000, random_state=42
        )

        # Step 2: Remove the random state before training
        result = repeated_cv_confidence_intervals(clone(best_model), X_train, y_train, n_repeats=1000, n_folds=5, ci=0.95)
        print("95% Confidence Interval for Train Score:", result)

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='binary')
        test_recall = recall_score(y_test, y_pred, average='binary')
        test_f1 = f1_score(y_test, y_pred, average='binary')

        # Combine all metrics
        evaluation_results.append({
            'Model': name,
            'Best CV Balanced_Accuracy': best_score,
            'p-value': p_value,
            'Test Accuracy': test_acc,
            'Test Balanced_Accuracy': test_bal_acc,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1 Score': test_f1,
            'Train Vaccinees': list(train_vaccinees.items()),
            'Test Vaccinees': list(test_vaccinees.items()),
            'Best Parameters': best_params,
        })

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


