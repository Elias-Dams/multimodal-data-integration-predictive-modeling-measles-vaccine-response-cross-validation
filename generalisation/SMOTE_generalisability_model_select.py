import pandas as pd
import os

# --- Define your criteria ---
vaccine = "Hepatitis" #Measles
dataname = "cytometry"
compression = "UNCOMPRESSED" # Or "COMPRESSED" as needed
results_file_path = f'../data_created/SMOTE/{vaccine}/TEST_{compression}_BALANCED_GENERAL_5000_SPLITS_{dataname}_data.csv'

# Overall performance criteria
test_ba_threshold = 0.5
train_acc_threshold = 0.70

# Classification report criteria based on Recall and Precision for each class
# Based on the discussion, aiming for performance at or above the 50/50 random baseline (0.5)
recall_0_threshold = 0.5
precision_0_threshold = 0.5
recall_1_threshold = 0.5
precision_1_threshold = 0.5

# --- Define column names ---
# Ensure these names match *exactly* the column headers in your CSV
test_ba_col = 'Test Balanced_Accuracy'
classification_report_col = 'Test Classification Report' # Column containing the classification report string
model_col = 'Model'
oversampling_col = 'Oversampling_Method'
train_acc_mean = 'CV mean' # Cross-validation mean accuracy
train_acc_median = 'CV median' # Cross-validation median accuracy
random_seed_col = 'Split seed' # Random_seed used for split

num_distinct_seeds_to_consider = 5000

# --- Script Logic ---
try:
    df_results = pd.read_csv(results_file_path)

    # --- Validate required columns ---
    required_cols = [model_col, oversampling_col, test_ba_col, random_seed_col,
                     train_acc_mean, train_acc_median, classification_report_col]
    if not all(col in df_results.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_results.columns]
        print(f"Error: The CSV file '{results_file_path}' is missing required columns: {', '.join(missing)}")
        exit()



    # 1. Get the first X distinct random seeds in the order they appear
    if random_seed_col not in df_results.columns:
        print(f"Error: The random seed column '{random_seed_col}' is not found in the CSV.")
        exit()

    # Get unique seeds in the order they first appear
    distinct_seeds_in_order = []
    for seed in df_results[random_seed_col]:
        if seed not in distinct_seeds_in_order:
            distinct_seeds_in_order.append(seed)
        if len(distinct_seeds_in_order) >= num_distinct_seeds_to_consider:
            break

    if len(distinct_seeds_in_order) < num_distinct_seeds_to_consider:
        print(
            f"Warning: Only {len(distinct_seeds_in_order)} distinct seeds available, requested {num_distinct_seeds_to_consider}.")
        # Use all available distinct seeds if fewer than requested
        first_x_distinct_seeds = distinct_seeds_in_order
    else:
        # Take only the first X distinct seeds
        first_x_distinct_seeds = distinct_seeds_in_order[:num_distinct_seeds_to_consider]

    # 2. Filter the DataFrame to include only rows with these specific seeds
    df_filtered_by_seeds = df_results[df_results[random_seed_col].isin(first_x_distinct_seeds)].copy()

    df_results = df_filtered_by_seeds

    # --- Apply filtering based on overall performance criteria ---
    # Start with models that meet the Balanced Accuracy and p-value thresholds
    filtered_df = df_results[
        (df_results[test_ba_col] >= test_ba_threshold) &
        (df_results[train_acc_mean] >= train_acc_threshold)
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
        display_cols = [model_col, oversampling_col, test_ba_col,
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

        print(len(selected_models_df.index.tolist()))

    else:
        print("\nNo models met all selection criteria.")

except FileNotFoundError:
    print(f"Error: The file '{results_file_path}' was not found. Please check the file path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")