
# Streamlit UI for interactive usage

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import tempfile
from src.utils.feature_extraction import extract_features

# Load pre-trained model
MODEL_PATH = "../models/alzheimer_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def predict_audio(file_path):
    """Predict Alzheimer's disease from an audio file."""
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Reshape for model input
    prediction = model.predict(features)
    result = "Alzheimer" if prediction[0][0] > 0.5 else "Healthy"
    return result, float(prediction[0][0])

# Streamlit UI
st.title("Alzheimer's Disease Prediction from Voice")
st.write("Upload an audio file to analyze whether the speaker may have Alzheimer's disease.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
    
    st.audio(uploaded_file, format='audio/wav')
    
    result, confidence = predict_audio(temp_path)
    st.write(f"### Prediction: {result}")
    st.write(f"### Confidence: {confidence:.4f}")


# Providing an interactive Streamlit UI for predicting Alzheimer's disease based on audio analysis. This script:

# Loads the pre-trained model (alzheimer_cnn.h5).
# Allows users to upload an audio file via a web interface.
# Extracts features and makes predictions with a confidence score.
# Displays the prediction and confidence interactively.




# Variant 2

# import streamlit as st
# from src.utils.audio_preprocessing import load_audio
# from src.models.train_model import load_trained_model

# model = load_trained_model("models/cnn_model.pth")

# st.title("Alzheimer Prediction Tool")

# uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
# if uploaded_file:
    # audio, sr = load_audio(uploaded_file)
    # prediction = model.predict(audio)
    # st.write(f"Prediction: {prediction*100:.2f}% likelihood of Alzheimer's.")