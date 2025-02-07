
# Model interpretability (SHAP, etc.)

import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.feature_extraction import extract_features

def load_model(model_path):
    """Load the trained model for explainability analysis."""
    return tf.keras.models.load_model(model_path)

def explain_prediction(model, input_audio):
    """Generate SHAP explanations for a given audio input."""
    features = extract_features(input_audio)
    features = np.expand_dims(features, axis=0)  # Reshape for SHAP analysis
    
    explainer = shap.Explainer(model, feature_names=[f"MFCC_{i}" for i in range(13)])
    shap_values = explainer(features)
    
    shap.summary_plot(shap_values, features)

def main():
    model_path = "../models/alzheimer_cnn.h5"
    test_audio = "../data/raw/sample.wav"
    
    print("Loading model...")
    model = load_model(model_path)
    
    print("Explaining prediction...")
    explain_prediction(model, test_audio)

if __name__ == "__main__":
    main()



# This script:

# Loads the trained model from ../models/alzheimer_cnn.h5.
# Extracts MFCC features from an audio input.
# Generates SHAP explanations to visualize how each feature (MFCC) impacts the model's predictions.
# Displays a summary plot of feature importance.