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
import ast
import numpy as np
import pandas as pd
import shap
import os

from exploration.SMOTE_utils import (
    handle_missing_values,
    load_groups_from_json,
    compress_correlated_features,
    encode_labels,
    split_dataset, scale_features, custom_stratified_metrics
)

###############################################################################
# CONFIGURATION
###############################################################################
LABELS = {'responder': {'name': 'response', 'color': 'blue'},
          'no response - high ab': {'name': 'no response', 'color': 'orange'},
          'no response - low ab':  {'name': 'no response', 'color': 'green'}
          }

configurations = [
    {
        "DATA_NAME": "RNa_data",
        "OVERSAMPLING_METHOD": None,
        "MODEL_NAME": "Naive Bayes",
        "COMPRESS_CORRELATED": True
    },
    {
        "DATA_NAME": "RNa_data",
        "OVERSAMPLING_METHOD": "smote",
        "MODEL_NAME": "SVM",
        "COMPRESS_CORRELATED": True
    },
    {
        "DATA_NAME": "RNa_data",
        "OVERSAMPLING_METHOD": "smote-borderline",
        "MODEL_NAME": "Logistic Regression",
        "COMPRESS_CORRELATED": True
    },
    {
        "DATA_NAME": "RNa_data",
        "OVERSAMPLING_METHOD": "smote-borderline",
        "MODEL_NAME": "SVM",
        "COMPRESS_CORRELATED": True
    },
    {
        "DATA_NAME": "RNa_data",
        "OVERSAMPLING_METHOD": "smote-adasyn",
        "MODEL_NAME": "SVM",
        "COMPRESS_CORRELATED": True
    },
    {
        "DATA_NAME": "RNa_data",
        "OVERSAMPLING_METHOD": "smote-smotetomek",
        "MODEL_NAME": "SVM",
        "COMPRESS_CORRELATED": True
    },
    {
        "DATA_NAME": "RNa_data",
        "OVERSAMPLING_METHOD": "smote-smoteenn",
        "MODEL_NAME": "Decision Tree",
        "COMPRESS_CORRELATED": True
    },
    {
        "DATA_NAME": "RNa_data",
        "OVERSAMPLING_METHOD": "smote-smoteenn",
        "MODEL_NAME": "Random Forest",
        "COMPRESS_CORRELATED": True
    },
]

#
# configurations = [
#     {
#         "DATA_NAME": "cytometry",
#         "OVERSAMPLING_METHOD": "smote",
#         "MODEL_NAME": "Naive Bayes",
#         "COMPRESS_CORRELATED": True
#     },
#     {
#         "DATA_NAME": "cytometry",
#         "OVERSAMPLING_METHOD": "smote-borderline",
#         "MODEL_NAME": "Naive Bayes",
#         "COMPRESS_CORRELATED": True
#     },
#     {
#         "DATA_NAME": "cytometry",
#         "OVERSAMPLING_METHOD": "smote-adasyn",
#         "MODEL_NAME": "Naive Bayes",
#         "COMPRESS_CORRELATED": True
#     },
#     {
#         "DATA_NAME": "cytometry",
#         "OVERSAMPLING_METHOD": "smote-smotetomek",
#         "MODEL_NAME": "Naive Bayes",
#         "COMPRESS_CORRELATED": True
#     },
#     {
#         "DATA_NAME": "cytometry",
#         "OVERSAMPLING_METHOD": "smote-smotetomek",
#         "MODEL_NAME": "Logistic Regression",
#         "COMPRESS_CORRELATED": True
#     }
# ]

all_shap_results = []

for configuration in configurations:

    DATA_NAME = configuration['DATA_NAME']
    OVERSAMPLING_METHOD = configuration['OVERSAMPLING_METHOD']
    MODEL_NAME = configuration['MODEL_NAME']
    COMPRESS_CORRELATED = configuration['COMPRESS_CORRELATED']

    SAVE_DIR = "../data_created/SMOTE/Measles/"              # Where to save CSV results
    compressed = "COMPRESSED" if COMPRESS_CORRELATED else "UNCOMPRESSED"
    SAVE_FILE = f"TEST_{compressed}_{DATA_NAME}_data.csv"
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

    abtiters = pd.read_csv('../data/Measles/antibody_df.csv')
    cytokines = pd.read_csv('../data/Measles/cytokines_data.csv')
    cytometry = pd.read_csv('../data/Measles/cyto_data.csv')
    clonal_breadth = pd.read_csv('../data/Measles/clonal_breadth_data.csv')
    clonal_depth = pd.read_csv('../data/Measles/clonal_depth_data.csv')
    module_scores = pd.read_csv('../data/Measles/RNA_circos.csv')

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

    datasets_current_model = datasets_merged[DATA_NAME]

    ###########################################################################
    # HANDLE MISSING VALUES
    ###########################################################################

    datasets_current_model["df"] = handle_missing_values(datasets_current_model["df"], DATA_NAME, datasets['antibody_titers'],strategy='mean')

    ###########################################################################
    # COMPRESS CORRELATED FEATURES
    ###########################################################################

    if COMPRESS_CORRELATED:
        groups = None
        if DATA_NAME == "cytokines":
            groups = load_groups_from_json("../data/Measles/clusters/cytokines.json")
        elif DATA_NAME == "cytometry":
            groups = load_groups_from_json("../data/Measles/clusters/cytometry.json")
        elif DATA_NAME == "RNa_data":
            groups = load_groups_from_json("../data/Measles/clusters/RNA1.json")
        # You can add additional elif blocks for "clonal_breadth" and "clonal_depth" if needed

        if groups is not None:
            datasets_current_model["df"] = compress_correlated_features(datasets_current_model["df"], groups)
        else:
            print(f"No compression groups defined for DATA_NAME = {DATA_NAME}")

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

    X_train, X_test, y_train, y_test = split_dataset(datasets_current_model["df"], train_vaccinees, test_vaccinees, oversampling_method=OVERSAMPLING_METHOD)
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

    selected_model_name = smote_df["Model"].iloc[0]
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

    metrics = custom_stratified_metrics(selected_model, X_train.copy(), y_train.copy(), cv_splits=5, random_state=42)

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

    # Check if the computed metrics match those in smote_df (within tolerance)
    if (abs(metrics['Accuracy'] - smote_df["Accuracy"].iloc[0]) < tol and
        abs(metrics['Balanced_Accuracy'] - smote_df["Balanced_Accuracy"].iloc[0]) < tol and
        abs(test_acc - smote_df["Test Accuracy"].iloc[0]) < tol and
        abs(test_bal_acc - smote_df["Test Balanced_Accuracy"].iloc[0]) < tol):
        print("!!!The computed metrics match those in smote_df!!!")
    else:
        print("!!!There is a discrepancy between the computed metrics and those in smote_df!!!")

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

    # shap.summary_plot(shap_values_[1], X_test, plot_type="bar")  # to the SHAP values for class 1
    # shap.summary_plot(shap_values_[1], X_test)  # Full SHAP summary plot

    threshold = 75

    # Create a dictionary to store the mean SHAP value for high feature values for each feature
    high_value_shap = {}

    # Loop over each feature (using its index and name)
    for i, feature in enumerate(X_test.columns):
        # Find indices where the feature value is above the threshold
        col_values = X_test[feature]
        scaled_values = (col_values - col_values.min()) / (col_values.max() - col_values.min()) * 100

        # Find the row numbers where the scaled feature value is above the threshold
        high_row_numbers = np.where(scaled_values > threshold)[0]

        # Only calculate if there are any samples meeting the threshold
        if len(high_row_numbers) > 0:
            # Extract the corresponding SHAP values for class 1 for this feature
            shap_for_feature = shap_values_[1][high_row_numbers, i]
            mean_shap = np.mean(shap_for_feature)
            high_value_shap[feature] = mean_shap
        else:
            high_value_shap[feature] = 0

    # Convert the results to a Pandas Series for easier interpretation
    high_value_shap_series = pd.Series(high_value_shap)

    mean_abs_shap = np.mean(np.abs(shap_values_[1]), axis=0)
    mean_abs_shap_series = pd.Series(mean_abs_shap, index=X_test.columns)
    print("Raw SHAP values (sorted):")
    print(mean_abs_shap_series.sort_values(ascending=False))

    # Normalize the SHAP values so that the maximum is 1
    normalized_shap_series = mean_abs_shap_series / mean_abs_shap_series.max()
    print("Normalized SHAP values (sorted):")
    print(normalized_shap_series.sort_values(ascending=False))

    current_shap_results = {
        "DATA_NAME": DATA_NAME,
        "OVERSAMPLING_METHOD": OVERSAMPLING_METHOD,
        "MODEL_NAME": MODEL_NAME,
        "mean_abs_shap": normalized_shap_series.to_dict(),
        "high_value_prediction_direction": high_value_shap_series.to_dict(),
    }

    all_shap_results.append(current_shap_results)

###########################################################################
# EVALUATING IMPORTANT FEATURES
###########################################################################

shap_df = pd.DataFrame([
    {
        "DATA_NAME": res["DATA_NAME"],
        "OVERSAMPLING_METHOD": res["OVERSAMPLING_METHOD"] if res["OVERSAMPLING_METHOD"] is not None else "None",
        "MODEL_NAME": res["MODEL_NAME"],
        **res["mean_abs_shap"]
    }
    for res in all_shap_results
])

directions_df = pd.DataFrame([
    {
        **res["high_value_prediction_direction"]
    }
    for res in all_shap_results
])


print("Aggregated SHAP results:")
print(shap_df.head())

# Get the feature columns (all columns except the configuration ones)
feature_cols = shap_df.columns.difference(["DATA_NAME", "OVERSAMPLING_METHOD", "MODEL_NAME"])

# Aggregate the normalized SHAP values across all configurations by computing the mean for each feature
aggregated_importance = shap_df[feature_cols].mean(axis=0)

# Also aggregate the direction values (they can be positive or negative)
aggregated_directions = directions_df[feature_cols].mean(axis=0)

# Combine magnitude with the direction (multiplying by the sign)
final_importance = aggregated_importance * np.sign(aggregated_directions)

# Sort the features in descending order by absolute importance
final_importance = final_importance.reindex(final_importance.abs().sort_values(ascending=True).index)

# Reorder the directions to match the final importance sorting
sorted_directions = aggregated_directions.reindex(final_importance.index)

colors = ['gray' if abs(val) < 0.01 else ('red' if val > 0 else 'dodgerblue') for val in sorted_directions]

print("Final aggregated feature importances (signed):")
print(final_importance)

# Plot a horizontal bar chart of the final feature importances
plt.figure(figsize=(10, 14))
final_importance.plot(kind="barh", color=colors)
plt.axvline(0, color="black", linewidth=1.0)  # Vertical line at zero
plt.title("Important Features (Aggregated Mean Normalized SHAP Values with Direction)")
plt.xlabel("Mean Normalized SHAP Value (Sign Indicates Prediction Direction)")
plt.ylabel("Feature")
# Force symmetric x-axis limits around zero:
current_xlim = plt.xlim()
max_val = max(abs(current_xlim[0]), abs(current_xlim[1]))
plt.xlim(-max_val, max_val)
xlims = plt.xlim()
ymin = plt.ylim()[0]
plt.text(xlims[0] + 0.05 * (xlims[1] - xlims[0]), ymin + 0.15, "Non Responder", color="blue", fontsize=12, ha="left")
plt.text(xlims[1] - 0.05 * (xlims[1] - xlims[0]), ymin + 0.15, "Responder", color="red", fontsize=12, ha="right")
plt.tight_layout()
plt.show()