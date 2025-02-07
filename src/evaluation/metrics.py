
# Performance metrics calculations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_metrics(y_true, y_pred):
    """Calculate classification performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc,
        "Confusion Matrix": conf_matrix.tolist()
    }
    
    return metrics

def main():
    test_results = pd.read_csv("../data/processed/test_results.csv")
    y_true = test_results["true_label"].values
    y_pred = test_results["predicted_label"].values
    
    metrics = evaluate_metrics(y_true, y_pred)
    
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()



# This script:

# Calculates key classification metrics:
# Accuracy, Precision, Recall, F1 Score, AUC (ROC), and Confusion Matrix.
# Loads test results from test_results.csv.
# Prints performance metrics to the console.
