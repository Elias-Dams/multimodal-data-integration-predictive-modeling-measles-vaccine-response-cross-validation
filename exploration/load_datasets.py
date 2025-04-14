import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch

LABELS = {'responder': {'name': 'response', 'color': 'blue'},
          'no response - high ab': {'name': 'no response', 'color': 'orange'},
          'no response - low ab':  {'name': 'no response', 'color': 'green'}
          }

###########################################################################
# Measles
###########################################################################

# All the data I use:
MEASLES_PATHS = {
    "antibody_titers": "../data/Measles/antibody_df.csv",
    "cytokines": "../data/Measles/cytokines_data.csv",
    "cytometry": "../data/Measles/cyto_data.csv",
    "clonal_breadth": "../data/Measles/clonal_breadth_data.csv",
    "clonal_depth": "../data/Measles/clonal_depth_data.csv",
    "RNA_data": "../data/Measles/RNA_circos.csv",
    "meta": "../data/Measles/metadata.csv"
}

def load_datasets_measles(add_metadate = False):

    # Load the antibody titers data (response profile)
    abtiters = pd.read_csv(MEASLES_PATHS["antibody_titers"])
    cytokines = pd.read_csv(MEASLES_PATHS["cytokines"])
    cytometry = pd.read_csv(MEASLES_PATHS["cytometry"])
    clonal_breadth = pd.read_csv(MEASLES_PATHS["clonal_breadth"])
    clonal_depth = pd.read_csv(MEASLES_PATHS["clonal_depth"])
    module_scores = pd.read_csv(MEASLES_PATHS["RNA_data"])
    meta = pd.read_csv(MEASLES_PATHS["meta"])
    meta.loc[:, 'Gender'] = meta['Gender'].map({'M': 1, 'F': 0})

    abtiters['response_label'] = abtiters['quadrant'].replace({key: value['name'] for key, value in LABELS.items()})

    datasets = {
        "antibody_titers": abtiters,
        "cytokines": cytokines,
        "cytometry": cytometry,
        "clonal_breadth": clonal_breadth,
        "clonal_depth": clonal_depth,
        "RNa_data": module_scores,
        **({"Meta": meta} if add_metadate else {})
    }

    print("Datasets loaded.")
    return datasets

def plot_labels_measles_facet(abtiters):
    # Get unique labels and count
    unique_labels = abtiters['response_label'].unique()
    n_labels = len(unique_labels)

    # Total subplots: one for each label plus one combined plot
    total_plots = n_labels
    if n_labels % 2 != 0:
        total_plots = n_labels + 1

    # Set maximum columns (e.g., 2) and compute required rows
    max_cols = 2
    n_rows = math.ceil(total_plots / max_cols)

    # Define evenly spaced x-axis positions and labels
    x_values = [0, 1, 2, 3]
    x_labels = ['Day 0', 'Day 21', 'Day 150', 'Day 365']

    # Create a figure with GridSpec for total_plots subplots
    fig = plt.figure(figsize=(6 * max_cols, 4.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, max_cols)

    # Flatten axes for easy indexing
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows * max_cols)]

    # Plot faceted subplots for each unique label in the first n_labels cells
    for idx, label in enumerate(unique_labels):
        ax = axes[idx]
        subset = abtiters[abtiters['response_label'] == label]
        n_lines = len(subset)
        # Generate colors from 'tab20' for each trajectory
        colors = plt.cm.tab20(np.linspace(0, 1, n_lines))
        for i, (_, row_data) in enumerate(subset.iterrows()):
            y_values = [row_data['Day 0'], row_data['Day 21'], row_data['Day 150'], row_data['Day 365']]
            ax.plot(x_values, y_values, marker='o', color=colors[i], alpha=0.8)
        ax.set_title(f"{label} (n={n_lines})")
        ax.set_xlabel("Time")
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Antibody Titer")
        ax.set_ylim(0, abtiters[['Day 0', 'Day 21', 'Day 150', 'Day 365']].max().max() + 10)
        ax.grid(True)

    if n_labels % 2 != 0:

        # Plot the combined plot in the next available cell (index = n_labels)
        ax_combined = axes[n_labels]
        # Define a fixed color mapping for each label (using tab10)
        fixed_colors = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}
        for label in unique_labels:
            subset = abtiters[abtiters['response_label'] == label]
            for _, row in subset.iterrows():
                y_values = [row['Day 0'], row['Day 21'], row['Day 150'], row['Day 365']]
                ax_combined.plot(x_values, y_values, marker='o', color=fixed_colors[label], alpha=0.8, label=label)
        ax_combined.set_title("Combined Antibody Titer Trajectories")
        ax_combined.set_xlabel("Time")
        ax_combined.set_xticks(x_values)
        ax_combined.set_xticklabels(x_labels)
        ax_combined.set_ylabel("Antibody Titer")
        ax_combined.set_ylim(0, abtiters[['Day 0', 'Day 21', 'Day 150', 'Day 365']].max().max() + 10)
        ax_combined.grid(True)
        # Create a legend with unique labels
        handles, labels = ax_combined.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_combined.legend(by_label.values(), by_label.keys(), title="Response Label")

    # Hide any unused axes
    for i in range(total_plots, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Antibody Titer Trajectories for Measles by Response Label (Quadrant)", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_labels_measles(abtiters):

    # Count the frequency of each label
    response_counts = abtiters['response_label'].value_counts()

    print("\nFrequency of responses:")
    print(response_counts)

    converted_dict = {value['name']: value['color'] for value in LABELS.values()}

    colors = [converted_dict[label] for label in response_counts.index]

    # Plot the frequency of high vs. low responses as a bar plot
    plt.figure(figsize=(8, 6))
    response_counts.plot(kind='bar', color=colors)
    plt.title('Frequency of High vs Low Titer Responses')
    plt.xlabel('Response Label')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(8, 6))
    for label, group in abtiters.groupby('response_label'):
        plt.scatter(group['Day 0'], group['Day 21'], color=converted_dict[label], label=label)

    # Plot reference line y = x
    plt.plot([0, max(abtiters['Day 0'])], [0, max(abtiters['Day 21'])], color='red', linestyle='-', linewidth=1)

    # Labeling and legend
    plt.xlabel('Day 0')
    plt.ylabel('Day 21')
    plt.legend(title="Titer Response")
    plt.show()

def merge_datasets_measles(datasets, add_metadate = False):
    datasets['antibody_titers'] = datasets['antibody_titers'].drop(columns=['vaccine', 'Day 0', 'Day 21', 'Day 150', 'Day 365', 'diff: 21-0', 'diff: 150-21', 'diff: 365-150','response', 'protected', 'quadrant'])
    datasets['cytometry'] = datasets['cytometry'][['Vaccinee', 'WBC Day 0', 'RBC Day 0', 'HGB Day 0', 'HCT Day 0', 'PLT Day 0', '%LYM Day 0', '%MON Day 0','%GRA Day 0']]

    cytokines_merged = pd.merge(datasets['antibody_titers'], datasets['cytokines'], on='Vaccinee')
    cytometry_merged = pd.merge(datasets['antibody_titers'], datasets['cytometry'], on='Vaccinee')
    clonal_breadth_merged = pd.merge(datasets['antibody_titers'], datasets['clonal_breadth'], on='Vaccinee')
    clonal_depth_merged = pd.merge(datasets['antibody_titers'], datasets['clonal_depth'], on='Vaccinee')
    rna_merged = pd.merge(datasets['antibody_titers'], datasets['RNa_data'], on='Vaccinee')

    datasets_merged = {
        "cytokines": {"df": cytokines_merged, "split": None},
        "cytometry": {"df": cytometry_merged, "split": None},
        "clonal_breadth": {"df": clonal_breadth_merged, "split": None},
        "clonal_depth": {"df": clonal_depth_merged, "split": None},
        "RNa_data": {"df": rna_merged, "split": None},
    }

    if add_metadate:
        meta_merged = pd.merge(datasets['antibody_titers'], datasets['Meta'], on='Vaccinee')
        datasets_merged["Meta"] = {"df": meta_merged, "split": None}


    print("Datasets merged.")
    return datasets_merged

def get_measles_data(visualise=True, add_metadate = False):
    # Load datasets
    datasets = load_datasets_measles(add_metadate = add_metadate)
    labels = datasets["antibody_titers"]

    if visualise:
        plot_labels_measles(labels)
        plot_labels_measles_facet(datasets["antibody_titers"])

    # Merge datasets
    datasets_merged = merge_datasets_measles(datasets, add_metadate = add_metadate)

    # Return merged datasets
    return datasets_merged, labels

###########################################################################
# Hepatitis B
###########################################################################

# All the data I use:
HEPATITIS_PATHS = {
    "antibody_titers": "../data/Hepatitis B/meta.csv",
    "classes": "../data/Hepatitis B/responder_classes.csv",
    "cytometry": "../data/Hepatitis B/cyto_data_corrected.csv",
    "TCR_predictions": "../data/Hepatitis B/TCRseq_pred_cv.csv",
    "RNA_data": "../data/Hepatitis B/RNA_circos.csv"
}

def get_response_label_hepatitis(row):
    """
    Determine the antibody response category by checking if the titres on Day 60,
    Day 180, and Day 365 are all above 12. If they are, label as 'responder';
    otherwise, label as 'no response - low ab'.
    """
    if (row['day60'] > 12) and (row['day180'] > 12) and (row['day365'] > 12):
        return 'responder'
    else:
        return 'no response - low ab'

def load_datasets_hepatitis(add_metadate = False):
    abtiters = pd.read_csv(HEPATITIS_PATHS["antibody_titers"])
    classes = pd.read_csv(HEPATITIS_PATHS["classes"])
    cytometry = pd.read_csv(HEPATITIS_PATHS["cytometry"])
    TCRseq_pred_cv = pd.read_csv(HEPATITIS_PATHS["TCR_predictions"])
    RNA_data = pd.read_csv(HEPATITIS_PATHS["RNA_data"])

    # Pivot the antibody titer data
    meta = abtiters[['Vaccinee', 'Gender', 'Age']]
    meta.loc[:, 'Gender'] = meta['Gender'].map({'M': 1, 'F': 0})

    abtiters = abtiters.pivot(index='Vaccinee', columns='Time_Point', values='Antibody_titre').reset_index()
    abtiters.columns = ['Vaccinee', 'day0', 'day60', 'day180', 'day365']
    abtiters = pd.merge(abtiters, classes, on='Vaccinee', how='left')

    # Calculate the response labels using the new criteria
    abtiters['response_label'] = abtiters.apply(get_response_label_hepatitis, axis=1)
    abtiters['response_label'] = abtiters['response_label'].replace({key: value['name'] for key, value in LABELS.items()})

    # Store datasets in a dictionary
    datasets = {
        "antibody_titers": abtiters,
        "cytometry": cytometry,
        "TCR_predictions": TCRseq_pred_cv,
        "RNa_data": RNA_data,
        **({"Meta": meta} if add_metadate else {})
    }

    print("Datasets loaded.")
    return datasets

def plot_labels_hepatitis(abtiters):
    abtiters_logged = abtiters.copy()

    # List the columns to log-scale
    cols_to_log = ['day0', 'day60', 'day180', 'day365']

    # Apply log10 transformation in-place
    for col in cols_to_log:
        abtiters_logged[col] = np.log10(abtiters[col])

    print(abtiters_logged[cols_to_log].head())

    # -------------------------------
    # Pivot Data to Long Format for Plotting
    # -------------------------------
    # For the conversion class colored plot, we'll include 'Vaccinee' and 'Class'
    df_long_class = pd.melt(abtiters_logged, id_vars=['Vaccinee', 'Class'],
                            value_vars=['day0', 'day60', 'day180', 'day365'],
                            var_name='Time_Point', value_name='Antibody_titre')

    # For the response label colored plot, we'll include 'Vaccinee' and 'response_label'
    df_long_resp = pd.melt(abtiters_logged, id_vars=['Vaccinee', 'response_label'],
                           value_vars=['day0', 'day60', 'day180', 'day365'],
                           var_name='Time_Point', value_name='Antibody_titre')

    # Map time point labels to numerical day values for plotting
    time_mapping = {'day0': 0, 'day60': 60, 'day180': 180, 'day365': 365}
    df_long_class['Time'] = df_long_class['Time_Point'].map(time_mapping)
    df_long_resp['Time'] = df_long_resp['Time_Point'].map(time_mapping)

    # -------------------------------
    # Define Color Mappings
    # -------------------------------
    # Color mapping based on Conversion Class
    class_colors = {
        'Early-converter': 'dodgerblue',
        'Late-converter': 'orangered',
        'Non-converter': 'limegreen'
    }

    # Color mapping based on Response Label
    response_colors = {
        'responder': 'green',
        'no response - high ab': 'blue',
        'no response - low ab': 'orange'
    }

    print("Color mapping by Conversion Class:")
    for key, val in class_colors.items():
        print(f"  {key}: {val}")

    print("\nColor mapping by Response Label:")
    for key, val in response_colors.items():
        print(f"  {key}: {val}")

    # -------------------------------
    # Plot 1: Trajectories Colored by Conversion Class
    # -------------------------------
    plt.figure(figsize=(10, 6))
    for vaccinee, group in df_long_class.groupby('Vaccinee'):
        group = group.sort_values('Time')
        # Each vaccinee is assumed to have one conversion class.
        conv_class = group['Class'].iloc[0]
        color = class_colors.get(conv_class, 'black')
        plt.plot(group['Time'], group['Antibody_titre'],
                 marker='o', linestyle='-', linewidth=1, color=color)

    plt.xlabel("Time (Days)")
    plt.ylabel("Antibody Titre")
    plt.title("Antibody Titre Trajectories (Colored by Conversion Class)\n(Selected Outliers Removed)")

    # Create a custom legend with only the three conversion classes.
    legend_elements_class = [
        Patch(facecolor='dodgerblue', label='Early-converter'),
        Patch(facecolor='orangered', label='Late-converter'),
        Patch(facecolor='limegreen', label='Non-converter')
    ]
    plt.legend(handles=legend_elements_class, title="Conversion Class", loc='upper right')
    plt.grid(True)
    plt.show()

    # -------------------------------
    # Plot 2: Trajectories Colored by Response Label
    # -------------------------------
    plt.figure(figsize=(10, 6))
    for vaccinee, group in df_long_resp.groupby('Vaccinee'):
        group = group.sort_values('Time')
        # Each vaccinee is assumed to have one response label.
        resp_label = group['response_label'].iloc[0]
        color = response_colors.get(resp_label, 'black')
        plt.plot(group['Time'], group['Antibody_titre'],
                 marker='o', linestyle='-', linewidth=1, color=color)

    plt.xlabel("Time (Days)")
    plt.ylabel("Antibody Titre")
    plt.title("Antibody Titre Trajectories (Colored by Response Label)\n(Selected Outliers Removed)")

    # Create a custom legend with only the three response labels.
    legend_elements_resp = [
        Patch(facecolor='green', label='responder'),
        Patch(facecolor='blue', label='no response - high ab'),
        Patch(facecolor='orange', label='no response - low ab')
    ]
    plt.legend(handles=legend_elements_resp, title="Response Label", loc='upper right')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))

    # Group the data by the "Class" column and plot each group with its corresponding color
    for converter_type, group in abtiters_logged.groupby("response_label"):
        plt.scatter(group["day0"], group["day60"],
                    color=response_colors.get(converter_type, "black"),
                    label=converter_type,
                    s=100, alpha=0.7)

    plt.xlabel("Day 0 Titer")
    plt.ylabel("Day 60 Titer")
    plt.title("Titer Response at Day 0 vs. Day 60\nColored by Converter Type")
    plt.legend(title="Converter Type")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    abtiters['response_label'] = abtiters['response_label'].replace({key: value['name'] for key, value in LABELS.items()})

    # --- Frequency Plot of Response Labels ---
    response_counts = abtiters['response_label'].value_counts()
    print("\nFrequency of responses:")
    print(response_counts)

    # Create a mapping from response label to color
    converted_dict = {value['name']: value['color'] for value in LABELS.values()}
    colors = [converted_dict[label] for label in response_counts.index]

    plt.figure(figsize=(8, 6))
    response_counts.plot(kind='bar', color=colors)
    plt.title('Frequency of High vs Low Titer Responses')
    plt.xlabel('Response Label')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()

def merge_datasets_hepatitis(datasets, add_metadate = False):
    datasets["antibody_titers"] = datasets["antibody_titers"].drop(columns=['day0', 'day60', 'day180', 'day365', 'Class'])
    datasets["cytometry"] = datasets["cytometry"][['Vaccinee', 'WBC Day 0', 'RBC Day 0', 'HGB Day 0', 'HCT Day 0', 'PLT Day 0', '%LYM Day 0', '%MON Day 0','%GRA Day 0']]

    cytometry_merged = pd.merge(datasets['antibody_titers'], datasets['cytometry'], on='Vaccinee')
    TCR_merged = pd.merge(datasets['antibody_titers'], datasets['TCR_predictions'], on='Vaccinee')
    rna_merged = pd.merge(datasets['antibody_titers'], datasets['RNa_data'], on='Vaccinee')

    datasets_merged = {
        "TCR_predictions": {"df": TCR_merged, "split": None},
        "cytometry": {"df": cytometry_merged, "split": None},
        "RNa_data": {"df": rna_merged, "split": None}
    }

    if add_metadate:
        meta_merged = pd.merge(datasets['antibody_titers'], datasets['Meta'], on='Vaccinee')
        datasets_merged["Meta"] = {"df": meta_merged, "split": None}

    print("Datasets merged.")
    return datasets_merged

def get_hepatitis_data(visualise=True, add_metadate=False):
    # Load datasets
    datasets = load_datasets_hepatitis(add_metadate = add_metadate)
    labels = datasets["antibody_titers"]

    if visualise:
        plot_labels_hepatitis(labels)

    # Merge datasets
    datasets_merged = merge_datasets_hepatitis(datasets, add_metadate = add_metadate)

    # Return merged datasets
    return datasets_merged, labels
