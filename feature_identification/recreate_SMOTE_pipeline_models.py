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
from collections import defaultdict

from helper_functions.utils import (
    handle_missing_values,
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

configurations = [498, 895, 899, 1282, 1349, 1395, 1396, 1733, 1738, 2096, 3681, 3686, 3720, 3722, 3791, 3873, 3915, 3920, 3995, 4440, 4497, 4920, 5320, 5368, 6521, 6931, 6936, 7069, 7198, 8756, 8761, 8997, 10048, 10116, 10124, 10549, 10846, 10956, 10961, 11269, 11936, 12523, 13123, 13848, 14066, 14206, 14221, 15741, 15997, 15998, 16216, 16445, 16448, 16816, 16990, 17320, 17340, 18916, 20296, 20448, 20720, 21416, 21446, 21733, 21738, 22932, 22937, 24121, 24122, 25497, 25620, 26666, 27023, 28274, 28598, 29245, 29765, 30674, 30765, 30834, 30839, 31106, 31111, 32623, 33270, 33297, 35022, 35041, 35446, 36095, 36248, 36423, 36432, 37081, 37086, 37096, 37370, 37721, 37849, 38908, 40522, 41758, 41763, 42095, 42440, 42566, 43097, 43672, 43895, 44040, 44091, 45491, 46418, 46641, 46972, 47696, 48243, 48573, 49246, 49696, 49808, 49813, 50055, 50060, 50981, 50983, 50986, 50988, 50995, 51024, 51441, 52743, 53095, 53096, 53647, 53821, 53998, 55295, 55358, 55363, 55748, 55847, 56524, 56596, 56772, 57870, 58116, 58121, 58790, 59145, 59245, 60341, 60641, 60648, 60716, 60865, 62018, 62744, 62966, 63381, 63386, 63395, 63971, 65895, 66948, 67406, 67411, 67596, 68924, 69224, 69273, 69284, 69289, 69299, 70606, 70611, 70621, 70841, 71021, 72390, 73465, 75316, 75473, 75641, 75721, 75897, 77465, 77798, 77846, 77848, 78741, 78891, 79066, 81866, 81946, 82182, 82690, 82968, 83268, 83416, 84767, 84790, 84793, 84946, 85495, 86073, 86256, 86261, 86271, 86748, 86873, 87320, 88141, 88148, 88891, 89483, 89488, 89640, 89670, 89673, 89945, 89948, 90781, 90786, 90897, 90898, 91156, 91161, 91673, 92115, 92299, 93516, 94199, 95131, 95136, 96031, 96036, 96632, 96646, 96940, 98861, 99530, 99535, 99821, 100420, 100424, 100631, 100636, 100646, 100974, 100995, 100998, 101995, 102016, 102121, 102193, 103841, 103931, 103936, 104196, 104522, 105021, 105422, 106181, 106186, 106215, 106243, 106891, 108649, 109018, 109765, 109781, 109786, 109791, 110016, 110240, 110421, 110873, 111183, 111188, 112198, 112398, 112474, 112530, 112535, 113166, 113420, 113666, 114720, 114793, 116241, 116356, 116361, 117148, 117891, 118255, 118260, 118371, 119217, 119272, 119616, 119699, 120406, 120421, 121731, 121736, 121845, 121847, 122541, 122597, 122973, 124070, 124890, 124893]

COMPRESS_CORRELATED = False
DATA_NAME = "cytometry"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

VISUALISE_DECICION_BOUNDRY = False
VISUALISE_OVERSAMPLING = False
VISUALISE_LABLES = False
VISUALISE_SHAP = False

VACCINE = "Hepatitis"
SAVE_DIR = f"../data_created/SMOTE/{VACCINE}/"
compressed = "COMPRESSED" if COMPRESS_CORRELATED else "UNCOMPRESSED"
SAVE_FILE = f"TEST_{compressed}_BALANCED_GENERAL_5000_SPLITS_{DATA_NAME}_data.csv"

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

check_missing_values(datasets_current_model["df"], DATA_NAME, len(abtiters))

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
                                                          random_state = RANDOM_STATE
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

    X_train = datasets_current_model["split"]["X_train"]
    X_test = datasets_current_model["split"]["X_test"]
    y_train = datasets_current_model["split"]["y_train"]
    y_test = datasets_current_model["split"]["y_test"]

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
    if MODEL_NAME not in models:
        raise ValueError(f"Selected model '{MODEL_NAME}' is not defined in our models dictionary.")

    # Get the model object and train it
    selected_model = clone(models[MODEL_NAME])

    metrics = custom_stratified_metrics(selected_model, X_train.copy(), y_train.copy(), cv_splits=5, random_state=RANDOM_STATE)

    # Train on full training data and calculate test accuracy
    selected_model.fit(X_train, y_train)

    if MODEL_NAME == "Naive Bayes":
        # If using GaussianNB, assign feature names to avoid warnings
        selected_model.feature_names_in_ = np.array(X_train.columns)

    y_pred = selected_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)

    print("Train Accuracy:", metrics['Accuracy'])
    print("Train Balanced Accuracy:", metrics['Balanced_Accuracy'])
    print("Test Accuracy:", test_acc)
    print("Test Balanced Accuracy:", test_bal_acc)

    # Set a tolerance for floating point comparisons
    tol = 1e-6

    if (abs(test_acc - current_df["Test Accuracy"].iloc[0]) < tol and
        abs(test_bal_acc - current_df["Test Balanced_Accuracy"].iloc[0]) < tol):
        print("!!!The computed metrics match those in current_df!!!")
    else:
        print("!!!There is a discrepancy between the computed metrics and those in current_df!!!")

    ###########################################################################
    # PCA & Decision Boundary Visualization
    ###########################################################################
    if VISUALISE_DECICION_BOUNDRY and PCA:
        visualize_decision_boundary_in_pca_space(
            selected_model,
            PCA,
            X_train,
            y_train,
            X_test,
            y_test,
            model_name = MODEL_NAME,
            data_name = DATA_NAME,
            methode_name = OVERSAMPLING_METHOD,
            save = True
        )

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
        if X_train.shape[0] < 100:
            background = X_train
        else:
            background = X_train.sample(n=100, random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model_predict, background, seed=RANDOM_STATE)

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

def plot_combined_shap(all_runs_data, TOP_N = 20):
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
    all_features = all_features[:TOP_N]
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
    global_min_x = 0
    global_max_x = 0

    elbow_plot_dots = []

    for f_idx, ax in enumerate(axes):
        data_list = points_by_feature[f_idx]  # list of (shap_val, scaled_val)
        shap_vals_array = np.array([p[0] for p in data_list])
        scaled_array = np.array([p[1] for p in data_list]) / 100.0  # map [0..100] => [0..1]

        # create vertical jitter
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

        elbow_plot_dots.append(aggregated_mean_abs_shap[feat_name])


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

    sorted_importance = sorted(elbow_plot_dots, reverse=True)
    ranks = np.arange(1, len(sorted_importance) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, sorted_importance, marker='o', linestyle='-')

    kl = KneeLocator(ranks, sorted_importance, curve='convex', direction='decreasing', S=1.0)
    elbow_rank_calculated = kl.elbow
    if elbow_rank_calculated is not None:
        elbow_importance_value_calculated = sorted_importance[elbow_rank_calculated - 1]
        plt.axvline(x=elbow_rank_calculated, color='red', linestyle='--',
                    label=f'Calculated Elbow at Rank {elbow_rank_calculated}')
        plt.plot(elbow_rank_calculated, elbow_importance_value_calculated, 'ro')  # Mark the elbow point

    plt.xlabel("Feature Rank (Sorted by Importance)")
    plt.ylabel("Aggregated Feature Importance Value")
    plt.title("Elbow Plot of Aggregated Feature Importance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

plot_combined_shap(all_shap_results, TOP_N = 30)