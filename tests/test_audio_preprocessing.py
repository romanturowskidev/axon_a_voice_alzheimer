
# Audio preprocessing

import os
import numpy as np
import librosa
import pytest
from src.utils.audio_preprocessing import preprocess_audio

def test_preprocess_audio():
    """Test if audio preprocessing correctly trims and normalizes audio."""
    input_file = "tests/sample_audio.wav"
    output_file = "tests/sample_audio_processed.wav"
    
    preprocess_audio(input_file, output_file)
    
    assert os.path.exists(output_file), "Output file was not created."
    
    y, sr = librosa.load(output_file, sr=16000)
    assert len(y) > 0, "Processed audio file is empty."
    assert np.all(np.abs(y) <= 1), "Audio is not normalized correctly."

def test_preprocess_audio_trim():
    """Test if silent parts are correctly trimmed from the audio file."""
    input_file = "tests/sample_silent_audio.wav"
    output_file = "tests/sample_trimmed_audio.wav"
    
    preprocess_audio(input_file, output_file)
    
    y_orig, _ = librosa.load(input_file, sr=16000)
    y_trimmed, _ = librosa.load(output_file, sr=16000)
    
    assert len(y_trimmed) < len(y_orig), "Trimming did not reduce audio length."

def test_preprocess_audio_invalid_file():
    """Test handling of an invalid input file."""
    with pytest.raises(Exception):
        preprocess_audio("invalid_file.wav", "output.wav")

if __name__ == "__main__":
    pytest.main()



# Test poprawności obróbki dźwięku (przycinanie, normalizacja),
# Test usuwania ciszy,
# Test obsługi błędów dla niepoprawnych plików.






# Variant 2

# from src.utils.audio_preprocessing import preprocess_audio

# def test_audio_preprocessing():
    # input_file = '../data/raw/sample.wav'
    # output_file = '../data/processed/sample.wav'
    # preprocess_audio(input_file, output_file)
    # assert os.path.exists(output_file)
