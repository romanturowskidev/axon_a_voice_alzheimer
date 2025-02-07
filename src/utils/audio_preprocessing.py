
# Audio cleaning & noise removal

import librosa
import librosa.effects
import numpy as np
import soundfile as sf

def preprocess_audio(input_path, output_path, sr=16000):
    """Load an audio file, trim silence, normalize, and save the processed file."""
    y, _ = librosa.load(input_path, sr=sr)
    y_trimmed, _ = librosa.effects.trim(y)
    y_normalized = librosa.util.normalize(y_trimmed)
    sf.write(output_path, y_normalized, sr)

def remove_noise(y, sr):
    """Apply noise reduction using spectral gating or other denoising techniques."""
    # Placeholder for noise reduction logic (can use scipy, noisereduce, etc.)
    return y  # Modify with real noise reduction algorithm

def extract_audio_features(input_path, sr=16000):
    """Extract raw audio signal and sample rate from a file."""
    y, sr = librosa.load(input_path, sr=sr)
    return y, sr

if __name__ == "__main__":
    sample_input = "../data/raw/sample.wav"
    sample_output = "../data/processed/sample_processed.wav"
    
    print("Processing audio...")
    preprocess_audio(sample_input, sample_output)
    print(f"Processed file saved: {sample_output}")


# This script:

# Loads an audio file and resamples it to 16kHz.
# Trims silence using librosa.effects.trim().
# Normalizes audio amplitude.
# Saves the processed audio in ../data/processed/.
# Placeholder for noise reduction, allowing further improvements.