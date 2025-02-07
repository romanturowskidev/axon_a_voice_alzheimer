
# Flask API for model inference

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from src.utils.feature_extraction import extract_features

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
MODEL_PATH = "../models/alzheimer_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for predicting Alzheimer's disease based on audio features."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    features = extract_features(file)
    features = np.expand_dims(features, axis=0)  # Reshape for model input
    prediction = model.predict(features)
    result = "Alzheimer" if prediction[0][0] > 0.5 else "Healthy"
    
    return jsonify({"prediction": result, "confidence": float(prediction[0][0])})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)



# This file contain the Flask-based API for predicting Alzheimer's disease based on audio features. This script:

# Loads the trained model from ../models/alzheimer_cnn.h5.
# Provides a /predict endpoint that accepts an audio file.
# Extracts features from the uploaded file and processes it through the model.
# Returns a JSON response with the prediction ("Alzheimer" or "Healthy") and its confidence score.





# variant 2

# from flask import Flask, request, jsonify
# from src.models.cnn_model import CNNModel

# app = Flask(__name__)
# model = CNNModel.load('../models/alzheimer_cnn.h5')

# @app.route('/predict', methods=['POST'])
# def predict():
    # audio_file = request.files['file']
    # features = extract_features(audio_file)
    # prediction = model.predict(features)
    # return jsonify({'prediction': prediction.tolist()})

# if __name__ == '__main__':
    # app.run(debug=True)