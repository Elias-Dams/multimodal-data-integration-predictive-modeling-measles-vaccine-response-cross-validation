import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_evaluation_results(evaluation_results, smote_method):
    """
    Plots a grouped bar plot of Train and Test Balanced Accuracy (with error bars)
    and annotates each model with its p-value. The title includes the oversampling method.

    Parameters:
    - evaluation_results: List of dictionaries with evaluation metrics.
    - smote_method: String indicating the oversampling method (e.g., 'smote', 'smote-borderline', etc.)
    """
    # Convert the list of dictionaries into a DataFrame.
    df_eval = pd.DataFrame(evaluation_results)

    # Reorder columns so that 'Train Vaccinees' and 'Test Vaccinees' are at the end.
    cols = list(df_eval.columns)
    cols_without = [col for col in cols if col not in ['Train Vaccinees', 'Test Vaccinees']]
    new_order = cols_without + ['Train Vaccinees', 'Test Vaccinees']
    df_eval = df_eval[new_order]

    # Calculate error bars for Train Balanced Accuracy.
    train_error_lower = df_eval['Train Balanced Accuracy'] - df_eval['Train IC lower']
    train_error_upper = df_eval['Train IC upper'] - df_eval['Train Balanced Accuracy']
    train_error = [np.abs(train_error_lower), np.abs(train_error_upper)]

    # Calculate error bars for Test Balanced Accuracy.
    test_error_lower = df_eval['Test Balanced Accuracy'] - df_eval['Test IC lower']
    test_error_upper = df_eval['Test IC upper'] - df_eval['Test Balanced Accuracy']
    test_error = [np.abs(test_error_lower), np.abs(test_error_upper)]

    # Create a grouped bar plot.
    models = df_eval['Model']
    x = np.arange(len(models))
    width = 0.35  # width of each bar

    fig, ax = plt.subplots(figsize=(12, 8))
    bars_train = ax.bar(x - width / 2, df_eval['Train Balanced Accuracy'], width,
                        yerr=train_error, capsize=5, label='Train Balanced Accuracy',
                        color='lightgreen', edgecolor='black')
    bars_test = ax.bar(x + width / 2, df_eval['Test Balanced Accuracy'], width,
                       yerr=test_error, capsize=5, label='Test Balanced Accuracy',
                       color='skyblue', edgecolor='black')

    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"Model Evaluation: Train and Test Balanced Accuracy with 95% CI\nSMOTE Method: {smote_method}",
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate each train bar with its value.
    for rect in bars_train:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    # Annotate each test bar with its value.
    for rect in bars_test:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    # Annotate each model with its p-value above the bars.
    for i, row in df_eval.iterrows():
        center_x = x[i]
        max_height = max(row['Train Balanced Accuracy'], row['Test Balanced Accuracy'])
        # Position annotation a little above the higher bar.
        ax.text(center_x, max_height + 0.05, f"p = {row['p-value']:.2f}",
                ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Optionally, print additional evaluation metrics for each model.
    for idx, row in df_eval.iterrows():
        print(f"\nModel: {row['Model']}")
        print(f"  Train Balanced Accuracy: {row['Train Balanced Accuracy']}")
        print(f"  Test Balanced Accuracy: {row['Test Balanced Accuracy']}")
        print(f"  p-value: {row['p-value']}")
        print(f"  95% Train CI: ({row['Train IC lower']}, {row['Train IC upper']})")
        print(f"  95% Test CI: ({row['Test IC lower']}, {row['Test IC upper']})")
        print(f"  Train Vaccinees: {row['Train Vaccinees']}")
        print(f"  Test Vaccinees: {row['Test Vaccinees']}")

    return df_eval

def plot_evaluation_results_by_smote(csv_file, output_dir):
    """
    Reads evaluation results from a CSV, groups them by the oversampling method,
    and produces a grouped bar plot for each oversampling method showing:
      - Train and Test Balanced Accuracy (with 95% CI error bars)
      - p-value annotations above the bars.

    Parameters:
    - csv_file: Path to the CSV file with evaluation results.
    - output_dir: Directory where the plots will be saved.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get unique oversampling methods from the dataframe
    smote_methods = df["Oversampling_Method"].unique()

    for method in smote_methods:
        # Check if method is NaN or an empty string
        if pd.isna(method) or method == "":
            df_method = df[df['Oversampling_Method'].isna() | (df['Oversampling_Method'] == "")].reset_index(drop=True)
        else:
            df_method = df[df["Oversampling_Method"] == method].reset_index(drop=True)

        # Set up the x positions and bar width
        models = df_method["Model"]
        x = np.arange(len(models))
        width = 0.35  # width of each bar

        # Compute error bars for Train Balanced Accuracy
        train_error_lower = df_method["CV mean"] - df_method["CV IC lower"]
        train_error_lower = train_error_lower.clip(lower=0)  # Ensure no negative error
        train_error_upper = df_method["CV IC upper"] - df_method["CV mean"]
        train_error_upper = train_error_upper.clip(lower=0)  # Ensure no negative error
        train_error = [train_error_lower, train_error_upper]

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width / 2, df_method["CV mean"], width,
                            yerr=train_error, capsize=5, label='Train Balanced Accuracy',
                            color='lightgreen', edgecolor='black')
        ax.bar(x + width / 2, df_method["Test Balanced_Accuracy"], width,
                            capsize=5, label='Test Balanced Accuracy',
                            color='skyblue', edgecolor='black')

        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"Model Evaluation: Balanced Accuracies\nOversampling Method: {method}", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        top_ylim = ax.get_ylim()[1]

        # Annotate each model with its p-value (positioned above the higher of the two bars)
        for i, row in df_method.iterrows():
            center_x = x[i]
            text_color = "green" if row["p-value"] <= 0.05 else "red"
            ax.text(center_x, top_ylim * 0.85, f"p = {row['p-value']:.2f}",
                    ha='center', va='bottom', fontsize=12, color=text_color, fontweight='bold')

        plt.tight_layout()
        data_name = csv_file.split("_", 2)[-1].replace(".csv", "")
        output_path = os.path.join(output_dir, f"evaluation_results_{data_name}_{method}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot for oversampling method '{method}' to {output_path}")



# Define your folder path(s) for each dataset's evaluation results
csv_files = [
    "../Measles/TEST_UNCOMPRESSED_clonal_breadth_data.csv",
    "../Measles/TEST_UNCOMPRESSED_clonal_depth_data.csv",
    "../Measles/TEST_UNCOMPRESSED_cytokines_data.csv",
    "../Measles/TEST_UNCOMPRESSED_cytometry_data.csv",
    "../Measles/TEST_UNCOMPRESSED_RNa_data_data.csv",
]

# Define the output directory where you want to save the plots
output_dir = "../Measles/uncompressed_plots"
os.makedirs(output_dir, exist_ok=True)

# Loop over all CSV files and generate plots
for csv_file in csv_files:
    plot_evaluation_results_by_smote(csv_file, output_dir)