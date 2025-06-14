import math

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to the training dataset.
    """
    smote_instance = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote_instance.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def train_model(X_train, y_train, model=None):
    """
    Train a Random Forest Classifier.
    """
    if model is None:
        raise ValueError('You have not added a model')
    rf_model = model
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_model(rf_model, X_test, y_test):
    """
    Evaluate the Random Forest Classifier using accuracy and confusion matrix.
    """
    y_pred = rf_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy_metrics = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    filtered_metrics = {'accuracy':accuracy_metrics['accuracy'],
                        'f1-macro avg':accuracy_metrics['macro avg']['f1-score'],
                        'f1-weighted avg': accuracy_metrics['weighted avg']['f1-score'],
                        'precision-macro avg': accuracy_metrics['macro avg']['precision'],
                        'precision-weighted avg': accuracy_metrics['weighted avg']['precision'],
                        }
    return filtered_metrics, conf_matrix


def shap_analysis(model, X_data, explainer=None, background_data=None):
    """
    Perform SHAP analysis for a trained model.
    """
    if explainer is None:
        raise ValueError('You have not added an explainer')

    explainer_instance = explainer(model.predict_proba, background_data, silent=True) if explainer == shap.KernelExplainer else explainer(model, background_data)
    shap_values = np.array(explainer_instance.shap_values(X_data, check_additivity=False))

    # Transpose to reorder the axes as needed.
    # found it on(https://stackoverflow.com/questions/65549588/shap-treeexplainer-for-randomforest-multiclass-what-is-shap-valuesi)
    shap_values_ = shap_values.transpose((2, 0, 1))  # Shape: (classes, samples, features)
    # shap_values__ = shap_values_.transpose((1, 0, 2))  # Shape: (samples, classes, features)

    return shap_values_


def plot_metrics(metrics_list, average_metric):
    """
    Plot accuracy, F1-macro avg, and F1-weighted avg per fold.
    """
    # Extract individual metrics across folds
    accuracies = [metrics['accuracy'] for metrics in metrics_list]
    f1_macro_avg = [metrics['f1-macro avg'] for metrics in metrics_list]
    f1_weighted_avg = [metrics['f1-weighted avg'] for metrics in metrics_list]
    precision_macro_avg = [metrics['precision-macro avg'] for metrics in metrics_list]
    precision_weighted_avg = [metrics['precision-weighted avg'] for metrics in metrics_list]

    # Number of folds
    n_folds = len(metrics_list)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each metric
    plt.plot(range(1, n_folds + 1), accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
    if average_metric == 'Macro' :
        plt.plot(range(1, n_folds + 1), f1_macro_avg, marker='o', linestyle='-', color='g', label='F1 Macro Avg')
        plt.plot(range(1, n_folds + 1), precision_macro_avg, marker='o', linestyle='-', color='r', label='Precision Macro Avg')
    else:
        plt.plot(range(1, n_folds + 1), f1_weighted_avg, marker='o', linestyle='-', color='g', label='F1 Weighted Avg')
        plt.plot(range(1, n_folds + 1), precision_weighted_avg, marker='o', linestyle='-', color='r', label='Precision Weighted Avg')

    # Plot average lines
    plt.axhline(y=np.mean(accuracies), color='b', linestyle='--', label=f'Average Accuracy = {np.mean(accuracies):.2f}')
    if average_metric == 'Macro':
        plt.axhline(y=np.mean(f1_macro_avg), color='g', linestyle='--',
                    label=f'Average F1 Macro Avg = {np.mean(f1_macro_avg):.2f}')
        plt.axhline(y=np.mean(precision_macro_avg), color='r', linestyle='--',
                    label=f'Average Precision Macro Avg = {np.mean(precision_macro_avg):.2f}')
    else:
        plt.axhline(y=np.mean(f1_weighted_avg), color='g', linestyle='--',
                    label=f'Average F1 Weighted Avg = {np.mean(f1_weighted_avg):.2f}')
        plt.axhline(y=np.mean(precision_weighted_avg), color='r', linestyle='--',
                    label=f'Average Precision Weighted Avg = {np.mean(precision_weighted_avg):.2f}')

    # Set titles and labels
    plt.title('Metrics per Fold (Accuracy, F1 Macro, F1 Weighted, Precision Macro Avg, Precision Weighted Avg)')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.xticks(range(1, n_folds + 1))
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.tight_layout()
    plt.show()


def interpret_conf_matrix(conf_matrix):
    """
    Interpret the confusion matrix in a more readable format.
    """
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"True Negatives (TN): {tn:.2f} -> Correctly predicted negative class.")
    print(f"False Negatives (FN): {fn:.2f} -> Incorrectly predicted negative class.")
    print(f"True Positives (TP): {tp:.2f} -> Correctly predicted positive class.")
    print(f"False Positives (FP): {fp:.2f} -> Incorrectly predicted positive class.")


def visualize_class_distribution(y, title="Class Distribution"):
    """
    Visualize the distribution of the target classes in the dataset.
    """
    if isinstance(y, list):
        # Dictionary to store class distributions and the corresponding fold numbers
        unique_distributions = {}

        for i, y_fold in enumerate(y):
            class_counts = pd.Series(y_fold).value_counts().sort_index()
            class_counts_tuple = tuple(class_counts)  # Convert the class counts to a tuple to use as a key

            if class_counts_tuple in unique_distributions:
                # If this distribution has been seen before, append the fold number
                unique_distributions[class_counts_tuple].append(i + 1)
            else:
                # Otherwise, create a new entry for this unique distribution
                unique_distributions[class_counts_tuple] = [i + 1]

        # Now plot the unique distributions
        n_plots = len(unique_distributions)
        fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 4), sharey=True)
        axes = axes if n_plots > 1 else [axes]

        for (class_counts_tuple, folds), ax in zip(unique_distributions.items(), axes):
            class_labels = range(len(class_counts_tuple))  # Class labels are inferred from the number of counts
            bars = ax.bar(class_labels, class_counts_tuple, color=['blue', 'orange'])
            # Split fold numbers into lines of up to 10 items
            fold_lines = [', '.join(map(str, folds[i:i + 8])) for i in range(0, len(folds), 8)]
            title_with_newlines = '\n'.join(fold_lines)

            ax.set_title(f"Folds \n{title_with_newlines}")
            ax.set_xlabel('Class Labels')
            ax.set_ylabel('Number of Samples')
            ax.set_xticks(class_labels)
            ax.grid(True)

            # Add exact values on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height - 2),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    else:
        # Single target variable (non-cross-validation case)
        class_counts = pd.Series(y).value_counts()
        class_labels = class_counts.index

        plt.figure(figsize=(2, 4))
        bars = plt.bar(class_labels, class_counts, color=['blue', 'orange'])
        plt.title(title)
        plt.xlabel('Class Labels')
        plt.ylabel('Number of Samples')
        plt.xticks(class_labels)
        plt.grid(True)

        # Add exact values on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width() / 2, height-2),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

        plt.show()


def model_with_shap(X, y, crossval="", n_splits=None, random_state=42, smote=True, model=None, shap_explainer=None, verbose=True):
    """
    Trains a model with optional cross-validation, performs SHAP analysis, and returns performance metrics.
    """

    results = {}

    def log(message):
        """Helper function for conditional logging based on verbose setting."""
        if verbose:
            print(message)

    if crossval == "K-fold":
        if n_splits is None:
            raise ValueError("K-fold cross-validation requires 'n_splits' to be set.")

        # Stratified K-Folds cross-validator
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Placeholder to accumulate SHAP values and other metrics
        shap_values_all_folds , train_labels_per_folds, accuracy_per_fold, conf_matrices = [], [], [], []

        # Split the dataset into training and test indices
        ix_training, ix_test = [], []
        for fold in skf.split(X, y):
            ix_training.append(fold[0]), ix_test.append(fold[1])

        # Cross-validation loop
        for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)):
            X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
            y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

            if smote:
                X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state=random_state)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            train_labels_per_folds.append(y_train_resampled)

            # Train Random Forest Classifier
            rf_model = train_model(X_train_resampled, y_train_resampled, model=model)

            # Predict and evaluate
            accuracy_metrics, conf_matrix = evaluate_model(rf_model, X_test, y_test)
            accuracy_per_fold.append(accuracy_metrics)
            conf_matrices.append(conf_matrix)

            # SHAP analysis
            shap_values = shap_analysis(rf_model, X_test, explainer=shap_explainer, background_data= X_train_resampled)
            shap_values_all_folds.append(shap_values[1])

        if smote:
            visualize_class_distribution(train_labels_per_folds, title="Class Distribution After SMOTE")
        else:
            visualize_class_distribution(train_labels_per_folds, title="Class Distribution")

        # Calculate and print average accuracy and confusion matrix
        if verbose:
            plot_metrics(accuracy_per_fold, "Weighted")

        average_conf_matrix = np.mean(conf_matrices, axis=0)
        interpret_conf_matrix(average_conf_matrix)

        new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
        shap.summary_plot(np.vstack(shap_values_all_folds), X.iloc[new_index])

    # Leave-One-Out Cross-Validation
    elif crossval == "Leave-One-Out":
        loo = LeaveOneOut()
        shap_values_all_folds, train_labels_per_folds, y_true, y_pred = [], [], [], []

        # Split the dataset into training and test indices
        ix_training, ix_test = [], []
        for fold in loo.split(X, y):
            ix_training.append(fold[0]), ix_test.append(fold[1])

        # Leave-One-Out Cross Validation loop
        for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)):
            X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
            y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

            if smote:
                X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state=random_state)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            # Store true labels for later evaluation
            y_true.append(y_test.values[0])

            # Store labels for later visualization
            train_labels_per_folds.append(y_train_resampled)

            # Train model
            rf_model = train_model(X_train_resampled, y_train_resampled, model=model)
            pred = rf_model.predict(X_test)
            y_pred.append(pred[0])

            shap_values = shap_analysis(rf_model, X_test, explainer=shap_explainer, background_data= X_train_resampled)
            shap_values_all_folds.append(shap_values[1])

        # Visualize the class distribution after SMOTE
        if verbose:
            if smote:
                visualize_class_distribution(train_labels_per_folds, title="Class Distribution After SMOTE")
            else:
                visualize_class_distribution(train_labels_per_folds, title="Class Distribution")

        # Calculate the metrics based on y_pred
        # make a conf_matrix
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        classification_report_values = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

        results = {
            'accuracy': accuracy,
            'f1_macro': classification_report_values['macro avg']['f1-score'],
            'f1_weighted': classification_report_values['weighted avg']['f1-score'],
            'precision_macro': classification_report_values['macro avg']['precision'],
            'precision_weighted': classification_report_values['weighted avg']['precision'],
            'conf_matrix': conf_matrix
        }

        if verbose:
            log(f"Accuracy: {results['accuracy']:.3f}")
            log(f"f1-macro avg: {results['f1_macro']:.3f}")
            log(f"f1-weighted avg: {results['f1_weighted']:.3f}")
            log(f"precision-macro avg: {results['precision_macro']:.3f}")
            log(f"precision-weighted avg: {results['precision_weighted']:.3f}")
            interpret_conf_matrix(conf_matrix)

            # Create a single index for SHAP summary plot across all iterations
            new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
            shap.summary_plot(np.vstack(shap_values_all_folds), X.iloc[new_index])

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        if smote:
            X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state=random_state)
            if verbose:
                visualize_class_distribution(y_train_resampled, title="Class Distribution After SMOTE")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
            if verbose:
                visualize_class_distribution(y_train_resampled, title="Class Distribution")

        rf_model = train_model(X_train_resampled, y_train_resampled, model=model)
        accuracy_metrics, conf_matrix = evaluate_model(rf_model, X_test, y_test)

        results = {
            'accuracy': accuracy_metrics['accuracy'],
            'f1_macro': accuracy_metrics['f1-macro avg'],
            'f1_weighted': accuracy_metrics['f1-weighted avg'],
            'precision_macro': accuracy_metrics['precision-macro avg'],
            'precision_weighted': accuracy_metrics['precision-weighted avg'],
            'conf_matrix': conf_matrix
        }

        if verbose:
            log(f"Accuracy: {results['accuracy']:.3f}")
            log(f"f1-macro avg: {results['f1_macro']:.3f}")
            log(f"f1-weighted avg: {results['f1_weighted']:.3f}")
            log(f"precision-macro avg: {results['precision_macro']:.3f}")
            log(f"precision-weighted avg: {results['precision_weighted']:.3f}")
            interpret_conf_matrix(conf_matrix)

            # SHAP analysis
            shap_values = shap_analysis(rf_model, X_test, explainer=shap_explainer, background_data= X_train_resampled)
            # Plot SHAP summary plot for class 1 (positive class)
            shap.summary_plot(shap_values[1], X_test)

