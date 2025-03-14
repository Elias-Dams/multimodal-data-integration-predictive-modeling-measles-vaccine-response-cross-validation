import json
import pandas as pd


# Load the CSV file
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


# Load the JSON file
def load_json(json_path):
    with open(json_path, 'r') as file:
        cluster_data = json.load(file)
    return cluster_data


# Check if every element in the cluster corresponds to a column in the CSV
def check_cluster_columns(csv_path, json_path):
    df = load_csv(csv_path)
    cluster_data = load_json(json_path)

    csv_columns = set(df.columns)
    cluster_elements = set()

    for cluster, elements in cluster_data.items():
        cluster_elements.update(elements)
        missing_elements = [element for element in elements if element not in csv_columns]
        if missing_elements:
            print(f"Cluster '{cluster}' has elements not in CSV columns: {missing_elements}")
        else:
            print(f"Cluster '{cluster}' is fully present in the CSV columns.")

    # Find columns that are not in any cluster
    extra_columns = csv_columns - cluster_elements
    if extra_columns:
        print(f"Columns in CSV that are not in any cluster: {extra_columns}")
    else:
        print("All columns in the CSV are present in at least one cluster.")


# Paths to files
csv_file = "data/Hepatitis B/RNA_circos.csv"
json_file = "data/Hepatitis B/clusters/RNA1.json"

# Run the check
check_cluster_columns(csv_file, json_file)
