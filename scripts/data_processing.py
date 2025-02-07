

# Plik zawiera kod do przetwarzania danych audio, w tym:

# Przygotowanie plików audio: przycinanie, normalizacja,
# Ekstrakcja cech (MFCC) dla uczenia maszynowego,
# Zapisywanie wyników w CSV.


import os
import librosa
import librosa.display
import numpy as np
import pandas as pd

def preprocess_audio(file_path, output_path):
    """Load, trim, and normalize an audio file before saving."""
    y, sr = librosa.load(file_path, sr=16000)
    y_trimmed, _ = librosa.effects.trim(y)
    y_normalized = librosa.util.normalize(y_trimmed)
    librosa.output.write_wav(output_path, y_normalized, sr)

def extract_features(file_path):
    """Extracts MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    return mfccs

def process_dataset(input_dir, output_dir, feature_csv):
    """Processes all audio files in a directory, extracts features, and saves results."""
    os.makedirs(output_dir, exist_ok=True)
    feature_list = []
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            
            preprocess_audio(input_path, output_path)
            features = extract_features(output_path)
            feature_list.append(features)
    
    df = pd.DataFrame(feature_list)
    df.to_csv(feature_csv, index=False)

if __name__ == "__main__":
    input_directory = '../data/raw/'
    output_directory = '../data/processed/'
    feature_file = '../data/processed/audio_features.csv'
    
    process_dataset(input_directory, output_directory, feature_file)









# import librosa
# import os

# def preprocess_audio(file_path, output_path):
    # y, sr = librosa.load(file_path, sr=16000)
    # y_trimmed, _ = librosa.effects.trim(y)
    # librosa.output.write_wav(output_path, y_trimmed, sr)

# Process all files
# input_dir = '../data/raw/'
# output_dir = '../data/processed/'
# for file in os.listdir(input_dir):
    # if file.endswith('.wav'):
        # preprocess_audio(os.path.join(input_dir, file), os.path.join(output_dir, file))



# Variant 2

# import librosa

# def load_audio(file_path, sr=16000):
    # return librosa.load(file_path, sr=sr)

# def trim_silence(audio, threshold=0.02):
    # return librosa.effects.trim(audio, top_db=threshold)
