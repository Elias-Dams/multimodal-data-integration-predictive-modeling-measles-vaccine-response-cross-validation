from joblib.parallel import method
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from kneed import KneeLocator
import ast
import numpy as np
import pandas as pd
import shap
import os

import matplotlib.colors as mcolors
import matplotlib.cm as cm

from helper_functions.utils import (
    load_groups_from_json,
    compress_correlated_features,
    encode_labels,
    split_dataset, scale_features, custom_stratified_metrics, visualize_decision_boundary_in_pca_space,
    check_missing_values
)
from helper_functions.load_datasets import get_measles_data, get_hepatitis_data

###############################################################################
# CONFIGURATION
###############################################################################

configurations = [5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24]

COMPRESS_CORRELATED = False
DATA_NAME = "cytometry"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

VISUALISE_DECICION_BOUNDARY = True
VISUALISE_OVERSAMPLING = True
VISUALISE_LABLES = False
VISUALISE_SHAP = False


SAVE_DIR = f"../data_created/SMOTE/Measles/"
OUTPUT_DIR = f"../data_created/VALIDATE/"
compressed = "COMPRESSED" if COMPRESS_CORRELATED else "UNCOMPRESSED"
SAVE_FILE = f"TEST_{compressed}_BALANCED_1_SPLIT_{DATA_NAME}_data.csv"

###########################################################################
# LOAD SMOTE DATA FILE
###########################################################################
file_path = os.path.join(SAVE_DIR, SAVE_FILE)

if os.path.exists(file_path):
    smote_df = pd.read_csv(file_path)
    print("Loaded DataFrame from:", file_path)
else:
    print(f"File not found: {file_path}")
    exit(0)

###########################################################################
# LOAD ALL THE DATA
###########################################################################

datasets_merged_Measles, abtiters = get_measles_data(visualise=VISUALISE_LABLES)
datasets_merged_Hepatitis, _ = get_hepatitis_data(visualise=VISUALISE_LABLES)

datasets_current_model = datasets_merged_Measles[DATA_NAME]  # the Measles dataset
validation_set = datasets_merged_Hepatitis[DATA_NAME]  # the hepatitis validation set

###########################################################################
# HANDLE MISSING VALUES
###########################################################################

check_missing_values(datasets_current_model["df"], DATA_NAME, len(abtiters))

###########################################################################
# COMPRESS CORRELATED FEATURES
###########################################################################
if COMPRESS_CORRELATED:
    groups_measles = None
    groups_hepatitis = None
    if DATA_NAME == "cytometry":
        groups_measles = load_groups_from_json("../data/Measles/clusters/cytometry.json")
        groups_hepatitis = load_groups_from_json("../data/Hepatitis B/clusters/cytometry.json")
    elif DATA_NAME == "RNa_data":
        groups_measles = load_groups_from_json("../data/Measles/clusters/RNA1.json")
        groups_hepatitis = load_groups_from_json("../data/Combined/clusters/RNA1.json")

    if groups_measles is not None and groups_hepatitis is not None:
        datasets_current_model["df"] = compress_correlated_features(datasets_current_model["df"], groups_measles)
        validation_set["df"] = compress_correlated_features(validation_set["df"], groups_hepatitis)
    else:
        print(f"No compression groups defined for DATA_NAME = {DATA_NAME}")

###########################################################################
# ENCODE THE LABELS (responder -> 1, non responder -> 0)
###########################################################################

datasets_current_model["df"] = encode_labels(datasets_current_model["df"])
validation_set["df"] = encode_labels(validation_set["df"])

all_shap_results = []
for index in configurations:

    ###############################################################################
    # FILTER THE DATAFRAME BASED INDEX
    ###############################################################################
    current_df = smote_df.loc[[index]]
    oversampling_value = current_df["Oversampling_Method"].iloc[0]
    if pd.isna(oversampling_value):
        OVERSAMPLING_METHOD = None
    else:
        OVERSAMPLING_METHOD = oversampling_value
    MODEL_NAME = current_df["Model"].iloc[0]

    ###########################################################################
    # SPLIT THE DATASET
    ###########################################################################

    def extract_vaccinee_ids(cell_value, id_index=1):
        try:
            # Convert the string to a Python object (list of tuples)
            tuples_list = ast.literal_eval(cell_value)
            # Extract the desired element from each tuple
            ids = [t[id_index] for t in tuples_list]
            return ids
        except Exception as e:
            print("Error extracting vaccine IDs:", e)
            return []

    test_vaccinees_str = current_df['Test Vaccinees'].iloc[0]
    test_vaccinees = extract_vaccinee_ids(test_vaccinees_str)
    print("Test Vaccinees:", test_vaccinees)

    train_vaccinees_str = current_df['Train Vaccinees'].iloc[0]
    train_vaccinees = extract_vaccinee_ids(train_vaccinees_str)
    print("Train Vaccinees:", train_vaccinees)

    X_train, X_test, y_train, y_test, PCA = split_dataset(datasets_current_model["df"], train_vaccinees, test_vaccinees,
                                                          oversampling_method=OVERSAMPLING_METHOD,
                                                          visualise_oversampling=VISUALISE_OVERSAMPLING,
                                                          model_name = MODEL_NAME,
                                                          data_name = DATA_NAME,
                                                          methode_name = OVERSAMPLING_METHOD,
                                                          )

    # Separate features and labels for the Hepatitis Validation set
    X_val = validation_set["df"].drop(['Vaccinee', 'response_label'], axis=1)
    y_val = validation_set["df"]['response_label']

    ###########################################################################
    # SCALE THE FEATURES
    ###########################################################################

    X_train_scaled, X_test_scaled, X_val_scaled = scale_features(X_train, X_test, X_val=X_val)

    ###########################################################################
    # TRAIN AND PREDICT MODEL
    ###########################################################################

    selected_model_name = current_df["Model"].iloc[0]
    print("Selected model from smote_df:", selected_model_name)

    # Compute balanced class weights from the training data
    classes = np.unique(y_train)
    balanced_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    custom_class_weights = dict(zip(classes, balanced_weights))

    # Define a dictionary of models
    models = {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight=custom_class_weights),
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight=custom_class_weights),
        "SVM": SVC(random_state=RANDOM_STATE, class_weight=custom_class_weights, probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=custom_class_weights),
        "Naive Bayes": GaussianNB()
    }

    # Check if the selected model name exists in our dictionary
    if selected_model_name not in models:
        raise ValueError(f"Selected model '{selected_model_name}' is not defined in our models dictionary.")

    # Get the model object and train it
    selected_model = clone(models[selected_model_name])

    metrics = custom_stratified_metrics(selected_model, X_train_scaled.copy(), y_train.copy(), cv_splits=5, random_state=RANDOM_STATE)

    # Train on full training data and calculate test accuracy
    selected_model.fit(X_train_scaled, y_train)

    if MODEL_NAME == "Naive Bayes":
        # If using GaussianNB, assign feature names to avoid warnings
        selected_model.feature_names_in_ = np.array(X_train_scaled.columns)

    y_pred = selected_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)

    y_pred_hepatitis = selected_model.predict(X_val_scaled)
    val_hepatitis_acc = accuracy_score(y_val, y_pred_hepatitis)
    val_hepatitis_bal_acc = balanced_accuracy_score(y_val, y_pred_hepatitis)

    print("Train Accuracy:", metrics['Accuracy'])
    print("Train Balanced Accuracy:", metrics['Balanced_Accuracy'])
    print("Test Accuracy:", test_acc)
    print("Test Balanced Accuracy:", test_bal_acc)
    print("Hepatitis Val Accuracy:", val_hepatitis_acc)
    print("Hepatitis Val Balanced Accuracy:", val_hepatitis_bal_acc)

    # Set a tolerance for floating point comparisons
    tol = 1e-6

    if (abs(test_acc - current_df["Test Accuracy"].iloc[0]) < tol and
        abs(test_bal_acc - current_df["Test Balanced_Accuracy"].iloc[0]) < tol):
        print("!!!The computed metrics match those in smote_df!!!")
    else:
        print("!!!There is a discrepancy between the computed metrics and those in smote_df!!!")

    ###########################################################################
    # PCA & Decision Boundary Visualization
    ###########################################################################
    if VISUALISE_DECICION_BOUNDARY and PCA:
        visualize_decision_boundary_in_pca_space(
            selected_model,
            PCA,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            model_name = MODEL_NAME,
            data_name = DATA_NAME,
            methode_name = OVERSAMPLING_METHOD,
            save = True
        )

    #check the decision boundary for the Hepatitis validations et
    if VISUALISE_DECICION_BOUNDARY and PCA:
        visualize_decision_boundary_in_pca_space(
            selected_model,
            PCA,
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            model_name = MODEL_NAME,
            data_name = DATA_NAME,
            methode_name = OVERSAMPLING_METHOD,
            save = True
        )

    ###########################################################################
    # Save output
    ###########################################################################

    result_entry = {
        'Model': MODEL_NAME,
        'Validation Accuracy': val_hepatitis_acc,
        'Validation Balanced_Accuracy': val_hepatitis_bal_acc,
    }

    results_df = pd.DataFrame([result_entry])

    results_df['Compress_Correlated'] = COMPRESS_CORRELATED
    results_df['Oversampling_Method'] = OVERSAMPLING_METHOD


    # Save evaluation results
    results_file = os.path.join(OUTPUT_DIR, SAVE_FILE)

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, mode='w', index=False)
        print(f"Created and saved evaluation results for {MODEL_NAME} to {results_file}")