import pandas as pd
import os


# --- Define your criteria ---
dataname = "RNa_data"
compression = "COMPRESSED" # Or "COMPRESSED" as needed
results_file_path = f'../data_created/SMOTE/Measles/TEST_{compression}_BALANCED_1_{dataname}_data.csv'

# Overall performance criteria
test_ba_threshold = 0.5
p_value_threshold = 0.05 # Assuming a p-value < 0.05 indicates statistically significant

# Classification report criteria based on Recall and Precision for each class
# Based on the discussion, aiming for performance at or above the 50/50 random baseline (0.5)
recall_0_threshold = 0.5
precision_0_threshold = 0.5
recall_1_threshold = 0.5
precision_1_threshold = 0.5

# --- Define column names ---
test_ba_col = 'Test Balanced_Accuracy'
p_value_col = 'p-value'
classification_report_col = 'Test Classification Report' # Column containing the classification report string
model_col = 'Model'
oversampling_col = 'Oversampling_Method'
train_acc_mean = 'CV mean' # Cross-validation mean accuracy
train_acc_median = 'CV median' # Cross-validation median accuracy


try:
    df_results = pd.read_csv(results_file_path)

    # --- Validate required columns ---
    required_cols = [model_col, oversampling_col, test_ba_col, p_value_col,
                     train_acc_mean, train_acc_median, classification_report_col]
    if not all(col in df_results.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_results.columns]
        print(f"Error: The CSV file '{results_file_path}' is missing required columns: {', '.join(missing)}")
        exit()

    # --- Apply filtering based on overall performance criteria ---
    # Start with models that meet the Balanced Accuracy and p-value thresholds
    filtered_df = df_results[
        (df_results[test_ba_col] >= test_ba_threshold) &
        (df_results[p_value_col] < p_value_threshold)
    ].copy()

    # --- Define a function to check classification report criteria ---
    def check_classification_report_criteria(report_string):
        """Parses a classification report string and checks if it meets metric thresholds."""
        if pd.isna(report_string):
            return False # Exclude models with no report

        try:
            report_dict = eval(report_string)

            # Check if expected class keys ('0', '1') exist
            if '0' not in report_dict or '1' not in report_dict:
                print(f"Warning: Classification report missing class keys '0' or '1': {report_string}")
                return False

            # Access metrics for class '0' and '1', provide default 0.0 if metric is missing
            recall_0 = report_dict.get('0', {}).get('recall', 0.0)
            precision_0 = report_dict.get('0', {}).get('precision', 0.0)
            recall_1 = report_dict.get('1', {}).get('recall', 0.0)
            precision_1 = report_dict.get('1', {}).get('precision', 0.0)

            # Check if all classification report criteria are met
            return (recall_0 >= recall_0_threshold and
                    precision_0 >= precision_0_threshold and
                    recall_1 >= recall_1_threshold and
                    precision_1 >= precision_1_threshold)

        except Exception as e:
            print(f"Warning: Could not parse classification report string: '{report_string}'. Error: {e}")
            return False # Exclude models with unparseable reports

    # --- Apply the classification report criteria to the filtered DataFrame ---
    selected_models_df = filtered_df.loc[
        filtered_df[classification_report_col].apply(check_classification_report_criteria)
    ].copy()

    # --- Display Results ---
    if not selected_models_df.empty:
        print("Selected models based on ALL criteria (Overall Performance + Classification Report Metrics):")
        # Define columns to display - include key metrics
        display_cols = [model_col, oversampling_col, test_ba_col, p_value_col,
                        train_acc_mean, train_acc_median, classification_report_col]
        display_cols = [col for col in display_cols if col in selected_models_df.columns]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_colwidth', 200)

        print(selected_models_df[display_cols])

        # -----------------------------------
        print("\nIndex of selected models:")
        print(selected_models_df.index.tolist())
        # -----------------------------------

    else:
        print("\nNo models met all selection criteria.")

except FileNotFoundError:
    print(f"Error: The file '{results_file_path}' was not found. Please check the file path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")