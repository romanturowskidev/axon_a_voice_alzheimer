
# Helper functions for ML models

import tensorflow as tf
import numpy as np
import os

def load_model(model_path):
    """Load a trained model from a given path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)

def predict(model, features):
    """Make a prediction using a trained model and extracted features."""
    features = np.expand_dims(features, axis=0)  # Reshape for model input
    prediction = model.predict(features)
    return prediction[0][0]

def save_model(model, model_path):
    """Save the trained model to a specified path."""
    model.save(model_path)
    print(f"Model saved successfully at {model_path}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return accuracy."""
    loss, accuracy = model.evaluate(X_test, y_test)
    return {"Loss": loss, "Accuracy": accuracy}

if __name__ == "__main__":
    model_path = "../models/alzheimer_cnn.h5"
    
    print("Loading model...")
    model = load_model(model_path)
    
    sample_features = np.random.rand(13)  # Simulated MFCC features
    prediction = predict(model, sample_features)
    print(f"Predicted Probability: {prediction:.4f}")
