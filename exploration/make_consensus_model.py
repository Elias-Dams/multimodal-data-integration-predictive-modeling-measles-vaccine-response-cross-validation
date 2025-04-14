import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score

# --- Define File Paths ---
BEST_MODELS_DIR = "../data_created/CONSENSUS/best_models"
TEST_SET_CSV_DIR = "../data_created/CONSENSUS/test_sets"

# configurations = {
#     "cytokines": {
#         'SAVE_FILE': 'TEST_UNCOMPRESSED',
#         'OVERSAMPLING_METHOD': 'random',
#         'BEST_MODELS_NAME': 'SVM',
#     },
#     "cytometry": {
#         'SAVE_FILE': 'TEST_UNCOMPRESSED',
#         'OVERSAMPLING_METHOD': 'smote',
#         'BEST_MODELS_NAME': 'Naive Bayes',
#     },
#     "clonal_breadth": {
#         'SAVE_FILE': 'TEST_UNCOMPRESSED',
#         'OVERSAMPLING_METHOD': 'smote',
#         'BEST_MODELS_NAME': 'Naive Bayes',
#     },
#     "clonal_depth": {
#         'SAVE_FILE': 'TEST_UNCOMPRESSED',
#         'OVERSAMPLING_METHOD': 'random',
#         'BEST_MODELS_NAME': 'SVM',
#     },
#     "RNa_data": {
#         'SAVE_FILE': 'TEST_UNCOMPRESSED',
#         'OVERSAMPLING_METHOD': 'smote',
#         'BEST_MODELS_NAME': 'Logistic Regression',
#     },
# }

configurations = {
    "cytokines": {
        'SAVE_FILE': 'TEST_COMPRESSED',
        'OVERSAMPLING_METHOD': 'random',
        'BEST_MODELS_NAME': 'Random Forest',
    },
    "cytometry": {
        'SAVE_FILE': 'TEST_COMPRESSED',
        'OVERSAMPLING_METHOD': 'random',
        'BEST_MODELS_NAME': 'Logistic Regression',
    },
    "clonal_breadth": {
        'SAVE_FILE': 'TEST_COMPRESSED',
        'OVERSAMPLING_METHOD': 'smote',
        'BEST_MODELS_NAME': 'Naive Bayes',
    },
    "clonal_depth": {
        'SAVE_FILE': 'TEST_COMPRESSED',
        'OVERSAMPLING_METHOD': 'smote',
        'BEST_MODELS_NAME': 'Random Forest',
    },
    "RNa_data": {
        'SAVE_FILE': 'TEST_COMPRESSED',
        'OVERSAMPLING_METHOD': 'random',
        'BEST_MODELS_NAME': 'SVM',
    },
}

loaded_models = []
model_names = []
all_predictions = {}
all_y_test = {}

# --- Load Models and Make Predictions on Their Respective Test Sets ---
for dataset in configurations:
    current_dataset = configurations[dataset]
    best_model_filename = os.path.join(
        BEST_MODELS_DIR,
        f"{current_dataset['SAVE_FILE']}_{dataset}_{current_dataset['BEST_MODELS_NAME']}_{current_dataset['OVERSAMPLING_METHOD']}_model.joblib"
    )
    test_set_csv_filename = os.path.join(
        TEST_SET_CSV_DIR,
        f"{current_dataset['SAVE_FILE']}_{dataset}_test_set.csv"
    )

    loaded_model = None
    try:
        loaded_model = joblib.load(best_model_filename)
        print(f"Model '{dataset}' loaded from: {best_model_filename}")
        loaded_models.append((dataset, loaded_model))
        model_names.append(dataset)
    except FileNotFoundError:
        print(f"Error: Model file not found for '{dataset}' at: {best_model_filename}")
    except Exception as e:
        print(f"Error loading model '{dataset}': {e}")

    test_df = None
    if os.path.exists(test_set_csv_filename):
        try:
            test_df = pd.read_csv(test_set_csv_filename)
            print(f"Test set for '{dataset}' loaded from CSV: {test_set_csv_filename}")
            if 'target' in test_df.columns:
                X_test_current = test_df.drop('target', axis=1)
                y_test_current = test_df['target'].values  # Get NumPy array for easier comparison later
                all_y_test[dataset] = y_test_current
                if loaded_model is not None:
                    try:
                        y_pred_current = loaded_model.predict(X_test_current)
                        all_predictions[dataset] = y_pred_current
                        print(f"Predictions made by '{dataset}' model.")
                    except Exception as e:
                        print(f"Error making predictions with '{dataset}' model: {e}")
            else:
                print(f"Error: 'target' column not found in test set for '{dataset}'.")
        except Exception as e:
            print(f"Error loading test set for '{dataset}': {e}")
    else:
        print(f"Error: Test set CSV file not found for '{dataset}'.")

# --- Perform Majority Voting ---
if all_predictions:
    # Ensure all prediction arrays have the same length (assuming same number of samples in test sets)
    first_dataset = next(iter(all_predictions))
    num_samples = len(all_predictions[first_dataset])
    final_predictions = np.zeros(num_samples, dtype=all_predictions[first_dataset].dtype)

    # Get the unique class labels (assuming they are the same across all y_test)
    unique_labels = np.unique(list(all_y_test.values())[0]) if all_y_test else None

    if unique_labels is not None:
        # Create an array to hold the votes for each sample and each class
        votes = np.zeros((num_samples, len(unique_labels)), dtype=int)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        for dataset, predictions in all_predictions.items():
            print(f"{predictions} ({dataset})")
            for i, pred in enumerate(predictions):
                if pred in label_to_index:
                    votes[i, label_to_index[pred]] += 1

        # Determine the final prediction by selecting the class with the most votes
        final_predictions = unique_labels[np.argmax(votes, axis=1)]

        print("\nFinal consensus predictions (majority voting):")
        print(final_predictions)

        # --- Evaluate the Consensus Predictions (against the labels of the first loaded test set) ---
        if all_y_test:
            first_y_test = list(all_y_test.values())[0]
            print(f"\nEvaluating consensus against the labels of the '{first_dataset}' test set:")
            accuracy_consensus = accuracy_score(first_y_test, final_predictions)
            balanced_accuracy_consensus = balanced_accuracy_score(first_y_test, final_predictions)
            report_consensus = classification_report(first_y_test, final_predictions)

            print(f"Consensus Accuracy: {accuracy_consensus:.4f}")
            print(f"Consensus Balanced Accuracy: {balanced_accuracy_consensus:.4f}")
            print("\nConsensus Classification Report:")
            print(report_consensus)
        else:
            print("Warning: No ground truth labels loaded for evaluation.")

    else:
        print("Error: Could not determine unique labels for majority voting.")

else:
    print("Error: No predictions were made. Cannot perform majority voting.")