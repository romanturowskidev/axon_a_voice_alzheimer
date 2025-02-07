
# Model training implementation

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.models.cnn_model import build_cnn_model

def load_data(file_path):
    """Load preprocessed dataset from a CSV file."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """Train the model on the training dataset."""
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return history

def main():
    """Load data, train CNN model, and save it."""
    dataset_path = "../data/processed/audio_features.csv"
    model_path = "../models/alzheimer_cnn.h5"
    
    print("Loading dataset...")
    X, y = load_data(dataset_path)
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Building CNN model...")
    model = build_cnn_model((X_train.shape[1], 1))
    
    print("Training model...")
    train_model(model, X_train, y_train, X_test, y_test)
    
    print("Saving trained model...")
    model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()


# This script:

# Loads preprocessed audio features from audio_features.csv.
# Splits data into training and testing sets.
# Builds a CNN model using cnn_model.py.
# Trains the model and evaluates it.
# Saves the trained model to alzheimer_cnn.h5.