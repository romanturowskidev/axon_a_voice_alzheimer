# Runs evaluation on test datasets.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_test_data(test_data_path):
    """Loads test dataset from a CSV file."""
    df = pd.read_csv(test_data_path)
    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values
    return X_test, y_test

def load_trained_model(model_path):
    """Loads the trained TensorFlow model."""
    return tf.keras.models.load_model(model_path)

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test dataset."""
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Alzheimer"]))

def main():
    test_data_path = '../data/processed/audio_features_test.csv'
    model_path = '../models/alzheimer_cnn.h5'
    
    print("Loading test dataset...")
    X_test, y_test = load_test_data(test_data_path)
    
    print("Loading trained model...")
    model = load_trained_model(model_path)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
if __name__ == "__main__":
    main()
