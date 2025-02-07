# Automates model training.

# Plik zawiera kod do:

# Ładowania i podziału zbioru danych,
# Tworzenia modelu przy użyciu sieci neuronowej,
# Trenowania modelu na danych treningowych,
# Ewaluacji modelu na danych testowych,
# Zapisywania modelu po zakończeniu treningu.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def load_data(file_path):
    """Loads preprocessed dataset from a CSV file."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_model(input_shape):
    """Defines and compiles a neural network model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """Trains the model on the training dataset."""
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the testing dataset."""
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    return loss, accuracy

def main():
    dataset_path = '../data/processed/audio_features.csv'
    model_path = '../models/alzheimer_cnn.h5'
    
    print("Loading dataset...")
    X, y = load_data(dataset_path)
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Building model...")
    model = build_model(X_train.shape[1])
    
    print("Training model...")
    train_model(model, X_train, y_train, X_test, y_test)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print("Saving model...")
    model.save(model_path)
    print(f"Model saved at {model_path}")
    
if __name__ == "__main__":
    main()







# Variant 2

# from src.models.cnn_model import CNNModel
# from src.utils.feature_extraction import load_features
# from sklearn.model_selection import train_test_split

# Load features
# X, y = load_features('../data/processed/audio_features.csv')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CNN
# model = CNNModel(input_shape=X_train.shape[1:])
# model.train(X_train, y_train)
# model.save('../models/alzheimer_cnn.h5')