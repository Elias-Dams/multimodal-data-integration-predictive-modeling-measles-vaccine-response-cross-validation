import joblib
import matplotlib.pyplot as plt
import ast
import numpy as np
import pandas as pd
import os

import matplotlib.colors as mcolors

from exploration.SMOTE_utils import (
    handle_missing_values,
    load_groups_from_json,
    compress_correlated_features,
    encode_labels,
    split_dataset, scale_features, custom_stratified_metrics, visualize_decision_boundary_in_pca_space
)
from exploration.load_datasets import get_measles_data, get_hepatitis_data


def visualize_decision_boundary_in_pca_space_from_saved(data_name, model_name, methode_name, X_train, y_train, X_test, y_test, n_points=200):
    """
    Visualize the decision boundary in PCA space using a saved model and PCA.
    This function loads the PCA and model from disk and projects the data into PCA space.
    Both the training and test data are plotted on the same decision boundary plot.
    """
    # Load saved PCA and model from disk
    pca = joblib.load(f'../data_created/SMOTE/saved_decision_boundaries/pca_{data_name}_{model_name}_{methode_name}.pkl')
    model = joblib.load(f'../data_created/SMOTE/saved_decision_boundaries/trained_{data_name}_{model_name}_{methode_name}.pkl')

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
    # Plot for Train and Test Data on the Same Plot
    # -------------------
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.linspace(Z.min(), Z.max(), 100))

    # Plot training data
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap, edgecolor="k", s=50, label="Train Data")

    # Plot test data
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap, edgecolor="k", s=50, marker="^", label="Test Data")

    # Title and labels
    plt.title(f"Decision Boundary (Train & Test) \n(Data: {data_name} Model: {model_name}, Methode: {methode_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()

configurations = [
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": None,
    #     "MODEL_NAME": "Random Forest",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": None,
    #     "MODEL_NAME": "Logistic Regression",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": None,
    #     "MODEL_NAME": "SVM",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": None,
    #     "MODEL_NAME": "Decision Tree",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": None,
    #     "MODEL_NAME": "Naive Bayes",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote",
    #     "MODEL_NAME": "Random Forest",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote",
    #     "MODEL_NAME": "Logistic Regression",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote",
    #     "MODEL_NAME": "SVM",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote",
    #     "MODEL_NAME": "Decision Tree",
    #     "COMPRESS_CORRELATED": True
    # },
    {
        "DATA_NAME": "cytometry",
        "OVERSAMPLING_METHOD": "smote",
        "MODEL_NAME": "Naive Bayes",
        "COMPRESS_CORRELATED": True
    },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-borderline",
    #     "MODEL_NAME": "Random Forest",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-borderline",
    #     "MODEL_NAME": "Logistic Regression",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-borderline",
    #     "MODEL_NAME": "SVM",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-borderline",
    #     "MODEL_NAME": "Decision Tree",
    #     "COMPRESS_CORRELATED": True
    # },
    {
        "DATA_NAME": "cytometry",
        "OVERSAMPLING_METHOD": "smote-borderline",
        "MODEL_NAME": "Naive Bayes",
        "COMPRESS_CORRELATED": True
    },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-adasyn",
    #     "MODEL_NAME": "Random Forest",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-adasyn",
    #     "MODEL_NAME": "Logistic Regression",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-adasyn",
    #     "MODEL_NAME": "SVM",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-adasyn",
    #     "MODEL_NAME": "Decision Tree",
    #     "COMPRESS_CORRELATED": True
    # },
    {
        "DATA_NAME": "cytometry",
        "OVERSAMPLING_METHOD": "smote-adasyn",
        "MODEL_NAME": "Naive Bayes",
        "COMPRESS_CORRELATED": True
    },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-smotetomek",
    #     "MODEL_NAME": "Random Forest",
    #     "COMPRESS_CORRELATED": True
    # },
    {
        "DATA_NAME": "cytometry",
        "OVERSAMPLING_METHOD": "smote-smotetomek",
        "MODEL_NAME": "Logistic Regression",
        "COMPRESS_CORRELATED": True
    },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-smotetomek",
    #     "MODEL_NAME": "SVM",
    #     "COMPRESS_CORRELATED": True
    # },
    # {
    #     "DATA_NAME": "cytometry",
    #     "OVERSAMPLING_METHOD": "smote-smotetomek",
    #     "MODEL_NAME": "Decision Tree",
    #     "COMPRESS_CORRELATED": True
    # },
    {
        "DATA_NAME": "cytometry",
        "OVERSAMPLING_METHOD": "smote-smotetomek",
        "MODEL_NAME": "Naive Bayes",
        "COMPRESS_CORRELATED": True
    }
]

for configuration in configurations:

    VACCINE = "Hepatitis"
    DATA_NAME = configuration['DATA_NAME']
    OVERSAMPLING_METHOD = configuration['OVERSAMPLING_METHOD']
    MODEL_NAME = configuration['MODEL_NAME']
    COMPRESS_CORRELATED = configuration['COMPRESS_CORRELATED']
    VISUALISE_DECICION_BOUNDRY = True
    VISUALISE_OVERSAMPLING = True
    VISUALISE_LABLES = False
    VISUALISE_SHAP = True

    SAVE_DIR = f"../data_created/SMOTE/{VACCINE}/"              # Where to save CSV results
    compressed = "COMPRESSED" if COMPRESS_CORRELATED else "UNCOMPRESSED"
    SAVE_FILE = f"TEST_{compressed}_COMBINED_CORR_{DATA_NAME}_data.csv"
    RANDOM_STATE = 42

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

    ###############################################################################
    # FILTER THE DATAFRAME BASED ON MODEL NAME AND OVERSAMPLING METHOD
    ###############################################################################
    if OVERSAMPLING_METHOD is None:
        # Look for rows where Oversampling_Method is empty or NaN
        smote_df = smote_df[(smote_df["Model"] == MODEL_NAME) & ((smote_df["Oversampling_Method"].isnull()) | (smote_df["Oversampling_Method"] == ""))]
    else:
        smote_df = smote_df[(smote_df["Model"] == MODEL_NAME) & (smote_df["Oversampling_Method"] == OVERSAMPLING_METHOD)]

    ###########################################################################
    # LOAD ALL THE DATA
    ###########################################################################

    if VACCINE == "Measles":
        datasets_merged, abtiters = get_measles_data(visualise = VISUALISE_LABLES)
    elif VACCINE == "Hepatitis":
        datasets_merged, abtiters = get_hepatitis_data(visualise = VISUALISE_LABLES)
    else:
        raise ValueError(f"VACCINE = {VACCINE} which is not an option")

    datasets_current_model = datasets_merged[DATA_NAME]

    ###########################################################################
    # HANDLE MISSING VALUES
    ###########################################################################

    datasets_current_model["df"] = handle_missing_values(datasets_current_model["df"], DATA_NAME, abtiters, strategy='mean')

    ###########################################################################
    # COMPRESS CORRELATED FEATURES
    ###########################################################################
    if COMPRESS_CORRELATED:
        if VACCINE == "Measles":
            groups = None
            if DATA_NAME == "cytokines":
                groups = load_groups_from_json("../data/Measles/clusters/cytokines.json")
            elif DATA_NAME == "cytometry":
                groups = load_groups_from_json("../data/Measles/clusters/cytometry.json")
            elif DATA_NAME == "RNa_data":
                groups = load_groups_from_json("../data/Measles/clusters/RNA1.json")

            if groups is not None:
                datasets_current_model["df"] = compress_correlated_features(datasets_current_model["df"], groups)
            else:
                print(f"No compression groups defined for DATA_NAME = {DATA_NAME}")
        elif VACCINE == "Hepatitis":
            groups = None
            if DATA_NAME == "cytometry":
                groups = load_groups_from_json("../data/Hepatitis B/clusters/cytometry.json")
            elif DATA_NAME == "RNa_data":
                groups = load_groups_from_json("../data/Combined/clusters/RNA1.json")

            if groups is not None:
                datasets_current_model["df"] = compress_correlated_features(datasets_current_model["df"], groups)
            else:
                print(f"No compression groups defined for DATA_NAME = {DATA_NAME}")
        else:
            raise ValueError(f"VACCINE = {VACCINE} which is not an option")

    ###########################################################################
    # ENCODE THE LABELS (responder -> 1, non responder -> 0)
    ###########################################################################

    datasets_current_model["df"] = encode_labels(datasets_current_model["df"])

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

    test_vaccinees_str = smote_df['Test Vaccinees'].iloc[0]
    test_vaccinees = extract_vaccinee_ids(test_vaccinees_str)
    print("Test Vaccinees:", test_vaccinees)

    train_vaccinees_str = smote_df['Train Vaccinees'].iloc[0]
    train_vaccinees = extract_vaccinee_ids(train_vaccinees_str)
    print("Train Vaccinees:", train_vaccinees)

    X_train, X_test, y_train, y_test, PCA = split_dataset(datasets_current_model["df"], train_vaccinees, test_vaccinees,
                                                          oversampling_method=OVERSAMPLING_METHOD,
                                                          visualise_oversampling=VISUALISE_OVERSAMPLING,
                                                          model_name = MODEL_NAME,
                                                          data_name = DATA_NAME,
                                                          methode_name = OVERSAMPLING_METHOD,
                                                          )
    split = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    datasets_current_model["split"] = split

    ###########################################################################
    # SCALE THE FEATURES
    ###########################################################################

    X_train_scaled, X_test_scaled = scale_features(datasets_current_model["split"]['X_train'], datasets_current_model["split"]['X_test'])
    datasets_current_model["split"]['X_train'] = X_train_scaled
    datasets_current_model["split"]['X_test'] = X_test_scaled

    ###########################################################################
    # TRAIN AND PREDICT MODEL
    ###########################################################################

    # Assume datasets_current_model["split"] is already defined
    X_train = datasets_current_model["split"]["X_train"]
    X_test = datasets_current_model["split"]["X_test"]
    y_train = datasets_current_model["split"]["y_train"]
    y_test = datasets_current_model["split"]["y_test"]

    ###########################################################################
    # CHECK DECISION BOUNDARY
    ###########################################################################

    visualize_decision_boundary_in_pca_space_from_saved(
        DATA_NAME,
        MODEL_NAME,
        OVERSAMPLING_METHOD,
        X_train,
        y_train,
        X_test,
        y_test,
    )