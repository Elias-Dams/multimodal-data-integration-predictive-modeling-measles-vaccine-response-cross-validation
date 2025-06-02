import json

data_name = "cytometry"
compress_correlated = True
model_names = ["Random Forest", "Logistic Regression", "SVM", "Decision Tree", "Naive Bayes"]
oversampling_methods = [None, "smote", "smote-borderline", "smote-adasyn", "smote-smotetomek", "smote-smoteenn"]

configurations = []
for model in model_names:
    for method in oversampling_methods:
        configurations.append({
            "DATA_NAME": data_name,
            "OVERSAMPLING_METHOD": method,
            "MODEL_NAME": model,
            "COMPRESS_CORRELATED": compress_correlated
        })

# Print the JSON structure
print(json.dumps(configurations, indent=4))