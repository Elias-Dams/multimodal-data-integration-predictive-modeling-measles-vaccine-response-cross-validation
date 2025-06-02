from joblib.parallel import method
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import ast
import numpy as np
import pandas as pd
import shap
import os

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from collections import defaultdict

from helper_functions.utils import (
    handle_missing_values,
    load_groups_from_json,
    compress_correlated_features,
    encode_labels,
    split_dataset, scale_features, custom_stratified_metrics, visualize_decision_boundary_in_pca_space, oversample_data
)
from helper_functions.load_datasets import get_measles_data, get_hepatitis_data

###############################################################################
# CONFIGURATION
###############################################################################


configurations = [
    {
        "DATA_NAME": "cytometry",
        "OVERSAMPLING_METHOD": None,
        "MODEL_NAME": "Logistic Regression",
        "COMPRESS_CORRELATED": False
    },
    {
        "DATA_NAME": "cytometry",
        "OVERSAMPLING_METHOD": None,
        "MODEL_NAME": "Logistic Regression",
        "COMPRESS_CORRELATED": False
    }
]

all_shap_results = []

for configuration in configurations:

    VACCINE = "Measles"
    DATA_NAME = configuration['DATA_NAME']
    OVERSAMPLING_METHOD = configuration['OVERSAMPLING_METHOD']
    MODEL_NAME = configuration['MODEL_NAME']
    COMPRESS_CORRELATED = configuration['COMPRESS_CORRELATED']
    VISUALISE_DECICION_BOUNDRY = False
    VISUALISE_OVERSAMPLING = False
    VISUALISE_LABLES = False
    VISUALISE_SHAP = True

    SAVE_DIR = f"../data_created/CONSENSUS/{VACCINE}/"              # Where to save CSV results
    compressed = "COMPRESSED" if COMPRESS_CORRELATED else "UNCOMPRESSED"
    SAVE_FILE = f"TEST3_{compressed}_{DATA_NAME}_data.csv"
    RANDOM_STATE = 42

    ###########################################################################
    # LOAD SMOTE DATA FILE
    ###########################################################################
    file_path = os.path.join(SAVE_DIR, SAVE_FILE)

    if os.path.exists(file_path):
        consensus_df = pd.read_csv(file_path)
        print("Loaded DataFrame from:", file_path)
    else:
        print(f"File not found: {file_path}")
        exit(0)

    ###############################################################################
    # FILTER THE DATAFRAME BASED ON MODEL NAME AND OVERSAMPLING METHOD
    ###############################################################################
    if OVERSAMPLING_METHOD is None:
        # Look for rows where Oversampling_Method is empty or NaN
        consensus_df = consensus_df[(consensus_df["Model"] == MODEL_NAME) & ((consensus_df["Oversampling_Method"].isnull()) | (consensus_df["Oversampling_Method"] == ""))]
    else:
        consensus_df = consensus_df[(consensus_df["Model"] == MODEL_NAME) & (consensus_df["Oversampling_Method"] == OVERSAMPLING_METHOD)]

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

    test_vaccinees_str = consensus_df['Test Vaccinees'].iloc[0]
    test_vaccinees = extract_vaccinee_ids(test_vaccinees_str)
    print("Test Vaccinees:", test_vaccinees)

    val_vaccinees_str = consensus_df['Validate Vaccinees'].iloc[0]
    val_vaccinees = extract_vaccinee_ids(val_vaccinees_str)
    print("Val Vaccinees:", val_vaccinees)

    train_vaccinees_str = consensus_df['Train Vaccinees'].iloc[0]
    train_vaccinees = extract_vaccinee_ids(train_vaccinees_str)
    print("Train Vaccinees:", train_vaccinees)

    train_df = datasets_current_model["df"][datasets_current_model["df"]['Vaccinee'].isin(train_vaccinees)]
    val_df = datasets_current_model["df"][datasets_current_model["df"]['Vaccinee'].isin(val_vaccinees)]
    test_df = datasets_current_model["df"][datasets_current_model["df"]['Vaccinee'].isin(test_vaccinees)]

    # Separate features and labels
    X_train = train_df.drop(['Vaccinee', 'response_label'], axis=1)
    y_train = train_df['response_label']
    X_val = val_df.drop(['Vaccinee', 'response_label'], axis=1)
    y_val = val_df['response_label']
    X_test = test_df.drop(['Vaccinee', 'response_label'], axis=1)
    y_test = test_df['response_label']

    X_train_resampled = X_train
    y_train_resampled = y_train

    if OVERSAMPLING_METHOD is not None:
        X_train_resampled, y_train_resampled = oversample_data(X_train, y_train, OVERSAMPLING_METHOD)

    split = {
        "X_train": X_train_resampled,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train_resampled,
        "y_val": y_val,
        "y_test": y_test,
    }
    datasets_current_model["split"] = split

    ###########################################################################
    # SCALE THE FEATURES
    ###########################################################################

    X_train = datasets_current_model["split"]['X_train']
    X_val = datasets_current_model["split"]['X_val']
    X_test = datasets_current_model["split"]['X_test']

    scaler = StandardScaler()
    # Fit and transform the training data, and transform the test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    # Wrap the numpy arrays back into DataFrames with the original indices and column names
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    datasets_current_model["split"]['X_train'] = X_train_scaled
    datasets_current_model["split"]['X_val'] = X_val_scaled
    datasets_current_model["split"]['X_test'] = X_test_scaled

    ###########################################################################
    # TRAIN AND PREDICT MODEL
    ###########################################################################

    X_train = datasets_current_model["split"]["X_train"]
    X_val = datasets_current_model["split"]["X_val"]
    X_test = datasets_current_model["split"]["X_test"]
    y_train = datasets_current_model["split"]["y_train"]
    y_val = datasets_current_model["split"]["y_val"]
    y_test = datasets_current_model["split"]["y_test"]

    selected_model_name = consensus_df["Model"].iloc[0]
    print("Selected model from consensus_df:", selected_model_name)

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

    # Train on full training data and calculate test accuracy
    selected_model.fit(X_train, y_train)
    y_val_pred = selected_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_bal_acc = balanced_accuracy_score(y_val, y_val_pred)

    if MODEL_NAME == "Naive Bayes":
        # If using GaussianNB, assign feature names to avoid warnings
        selected_model.feature_names_in_ = np.array(X_train.columns)

    # Train the best model on the combined training and validation data
    X_train = pd.concat([X_train, X_val], ignore_index=True)
    y_train = pd.concat([y_train, y_val], ignore_index=True)
    selected_model.fit(X_train, y_train)

    y_pred = selected_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)

    print("Val Accuracy:", val_acc)
    print("val Balanced Accuracy:", val_bal_acc)
    print("Test Accuracy:", test_acc)
    print("Test Balanced Accuracy:", test_bal_acc)

    # Set a tolerance for floating point comparisons
    tol = 1e-6

    if (abs(val_bal_acc - consensus_df["Validation Balanced Accuracy"].iloc[0]) < tol  and
        abs(test_acc - consensus_df["Test Accuracy"].iloc[0]) < tol and
        abs(test_bal_acc - consensus_df["Test Balanced_Accuracy"].iloc[0]) < tol):
        print("!!!The computed metrics match those in consensus_df!!!")
    else:
        print("!!!There is a discrepancy between the computed metrics and those in consensus_df!!!")

    ###########################################################################
    # PCA & Decision Boundary Visualization
    ###########################################################################
    # if VISUALISE_DECICION_BOUNDRY and PCA:
    #     visualize_decision_boundary_in_pca_space(
    #         selected_model,
    #         PCA,
    #         X_train,
    #         y_train,
    #         X_test,
    #         y_test,
    #         model_name = MODEL_NAME,
    #         data_name = DATA_NAME,
    #         methode_name = OVERSAMPLING_METHOD,
    #         save = True
    #     )

    ###########################################################################
    # ADDING SHAP EXPLANATIONS
    ###########################################################################

    if MODEL_NAME in ["Random Forest", "Decision Tree"]:
        # For tree-based models, use the TreeExplainer directly.
        explainer = shap.TreeExplainer(selected_model)
    else:
        def model_predict(data):
            # Ensure data is a DataFrame with the correct columns.
            data_df = pd.DataFrame(data, columns=X_train.columns)
            return selected_model.predict_proba(data_df)

        # Use a background sample from X_train (if too many rows, take the first 100)
        background = X_train if X_train.shape[0] < 100 else X_train.iloc[:100]
        explainer = shap.KernelExplainer(model_predict, background)

    # Calculate SHAP values on the test set.
    shap_values = np.array(explainer.shap_values(X_test, check_additivity=False))
    shap_values_ = shap_values.transpose((2, 0, 1))

    if VISUALISE_SHAP:
        # shap.summary_plot(shap_values_[1], X_test, plot_type="bar")  # to the SHAP values for class 1
        shap.summary_plot(shap_values_[1], X_test)  # Full SHAP summary plot

    mean_abs_shap = np.mean(np.abs(shap_values_[1]), axis=0)
    mean_abs_shap_series = pd.Series(mean_abs_shap, index=X_test.columns)

    # Normalize the SHAP values so that the maximum is 1
    normalized_shap_series = mean_abs_shap_series / mean_abs_shap_series.max()

    def normalize_columns_0_to_100(df: pd.DataFrame) -> pd.DataFrame:
        df_norm = df.copy()
        for col in df_norm.columns:
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()
            if col_max != col_min:
                df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min) * 100
            else:
                # Column is constant; just set it to 0
                df_norm[col] = 0
        return df_norm

    col_index_mapping = {}
    for idx, col in enumerate(X_test.columns):
        col_index_mapping[idx] = col

    current_shap_results = {
        "mapping": col_index_mapping,
        "mean_abs_shap": normalized_shap_series.to_dict(),
        "shap_values": shap_values_[1],
        "feature_values_scaled": normalize_columns_0_to_100(X_test).to_numpy(),
    }

    all_shap_results.append(current_shap_results)

###########################################################################
# EVALUATING IMPORTANT FEATURES
###########################################################################

def plot_combined_shap(all_runs_data):
    """
    Similar to plot_combined_shap, but uses a single shared x-axis across all subplots,
    so 0.0 is aligned for each feature. Also places one tall colorbar on the right.

    all_runs_data is a list of dicts, each with:
      {
        'shap_values': np.array of shape (n_samples, n_features),
        'feature_values_scaled': np.array of shape (n_samples, n_features) in [0..100],
        'mapping': dict {feature_index -> feature_name},
        'mean_abs_shap': dict {feature_name -> float}
      }
    """

    # ------------------------------------------------------
    # 1) AGGREGATE mean_abs_shap ACROSS RUNS
    # ------------------------------------------------------
    from collections import defaultdict
    aggregated_dict = defaultdict(list)

    for run_data in all_runs_data:
        for feat_name, val in run_data['mean_abs_shap'].items():
            aggregated_dict[feat_name].append(val)

    aggregated_mean_abs_shap = {}
    for feat_name, val_list in aggregated_dict.items():
        aggregated_mean_abs_shap[feat_name] = np.median(val_list)

    # Sort features by descending aggregated importance
    all_features = list(aggregated_mean_abs_shap.keys())
    all_features.sort(key=lambda f: aggregated_mean_abs_shap[f], reverse=True)
    feature_to_global_idx = {f: i for i, f in enumerate(all_features)}

    # ------------------------------------------------------
    # 2) COMBINE SHAP/FEATURE VALUES ACROSS RUNS
    # ------------------------------------------------------
    all_points = []  # list of (global_idx, shap_val, scaled_val)

    for run_data in all_runs_data:
        shap_vals = run_data['shap_values']  # shape (n_samples, n_features_run)
        scaled_vals = run_data['feature_values_scaled']  # shape (n_samples, n_features_run)
        mapping = run_data['mapping']  # {local_index -> feature_name}

        # invert the mapping: feature_name -> local_index
        inv_map = {name: idx for idx, name in mapping.items()}

        # for each feature in our global list
        for feat_name in all_features:
            if feat_name not in inv_map:
                continue
            local_idx = inv_map[feat_name]
            g_idx = feature_to_global_idx[feat_name]

            col_shap = shap_vals[:, local_idx]
            col_scaled = scaled_vals[:, local_idx]
            for i_sample in range(len(col_shap)):
                shap_val = col_shap[i_sample]
                scaled_val = col_scaled[i_sample]
                all_points.append((g_idx, shap_val, scaled_val))

    # group points by feature index
    n_features = len(all_features)
    points_by_feature = [[] for _ in range(n_features)]
    for (feat_idx, shap_val, scaled_val) in all_points:
        points_by_feature[feat_idx].append((shap_val, scaled_val))

    # ------------------------------------------------------
    # 3) CREATE SUBPLOTS (one row per feature), SHARE X
    # ------------------------------------------------------
    fig, axes = plt.subplots(n_features, 1,
                             figsize=(9, max(7, n_features * 0.5)),
                             sharex=True)  # share the same x-axis range

    # If there's only 1 feature, axes is not an array
    if n_features == 1:
        axes = [axes]

    colors = [
        "#008AFB",  # Blue (low)
        "#A719A8",  # Purple (mid)
        "#FF0A58"  # Pinkish Red (high)
    ]

    # 2) Create a custom LinearSegmentedColormap from the color list
    cmap = mcolors.LinearSegmentedColormap.from_list("my_rdbu", colors, N=256)  # reversed so 0=blue, 100=red

    # We'll track min/max shap value for setting x-lims manually if we want
    # but we can also let matplotlib auto-scale it across all features
    global_min_x = 0
    global_max_x = 0

    for f_idx, ax in enumerate(axes):
        data_list = points_by_feature[f_idx]  # list of (shap_val, scaled_val)
        shap_vals_array = np.array([p[0] for p in data_list])
        scaled_array = np.array([p[1] for p in data_list]) / 100.0  # map [0..100] => [0..1]

        # create vertical jitter
        # y_values = [0] * len(shap_vals_array)
        seen = {}
        y_values = []
        for val in shap_vals_array:
            rounded_val = round(val, 3)
            if rounded_val in seen:
                seen[rounded_val] += 0.001
            else:
                seen[rounded_val] = 0.0
            y_values.append(seen[rounded_val])

        # track global min/max for possible manual x-lim
        this_min = shap_vals_array.min()
        this_max = shap_vals_array.max()
        if this_min < global_min_x:
            global_min_x = this_min
        if this_max > global_max_x:
            global_max_x = this_max

        # plot
        ax.axvline(0, color="gray", linewidth=0.8)
        sc = ax.scatter(shap_vals_array, y_values, c=scaled_array, cmap=cmap,
                        alpha=0.8, s=20, edgecolor='k', linewidth=0.3)

        ax.set_yticks([])
        feat_name = all_features[f_idx]
        ax.set_ylabel(f"{feat_name} ({aggregated_mean_abs_shap[feat_name]:.4f})", rotation=0, labelpad=40, ha='right', va='center')

        # style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # hide x labels for all but bottom subplot
        if f_idx < n_features - 1:
            ax.set_xticklabels([])


    # Add a single x label at the bottom
    axes[-1].set_xlabel("SHAP Value")

    fig.suptitle("Combined SHAP Dot Plot (Shared X‐Axis)\nFeatures sorted by aggregated importance", y=0.98)

    # Adjust layout
    fig.tight_layout(rect=[0, 0.0, 0.86, 0.95])  # leave space on right for colorbar

    # ------------------------------------------------------
    # 4) COLORBAR on the RIGHT
    # ------------------------------------------------------
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.90, 0.05, 0.02, 0.9])  # adjust as needed
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", cax=cbar_ax,
                        fraction=0.02, pad=0.02, aspect=50)
    cbar.set_label("Scaled Feature Value (0 → 100)", rotation=90)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "50", "100"])

    cbar.outline.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    plt.show()

plot_combined_shap(all_shap_results)

