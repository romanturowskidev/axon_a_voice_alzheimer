# Feature extraction

import os
import numpy as np
import librosa
import pytest
from src.utils.feature_extraction import extract_features

def test_extract_features():
    """Test if feature extraction correctly extracts MFCC features from audio."""
    input_file = "tests/sample_audio.wav"
    
    features = extract_features(input_file)
    
    assert isinstance(features, np.ndarray), "Extracted features should be a NumPy array."
    assert features.shape == (13,), "Feature vector should have 13 MFCC coefficients."

def test_extract_features_invalid_file():
    """Test handling of an invalid input file."""
    with pytest.raises(Exception):
        extract_features("invalid_file.wav")

if __name__ == "__main__":
    pytest.main()


# Test poprawności ekstrakcji MFCC (sprawdza, czy funkcja zwraca poprawny wektor cech),
# Test obsługi błędów dla niepoprawnych plików audio.