
# Model validation before deployment

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.feature_extraction import extract_features

def load_model(model_path):
    """Load the trained model for validation."""
    return tf.keras.models.load_model(model_path)

def validate_model(model, test_data_path):
    """Validate the model using a test dataset."""
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.iloc[:, :-1].values
    y_true = test_data.iloc[:, -1].values
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }
    
    return metrics

def main():
    model_path = "../models/alzheimer_cnn.h5"
    test_data_path = "../data/processed/test_results.csv"
    
    print("Loading model...")
    model = load_model(model_path)
    
    print("Validating model...")
    metrics = validate_model(model, test_data_path)
    
    print("Validation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()



# This script:

# Loads the trained model from ../models/alzheimer_cnn.h5.
# Reads the test dataset from test_results.csv.
# Evaluates model performance using accuracy, precision, recall, and F1-score.
# Prints the validation results for final checks before deployment.