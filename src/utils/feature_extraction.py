

# Extracting MFCC, chroma, etc.

import librosa
import numpy as np
import pandas as pd

def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    """Extracts MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def extract_chroma(file_path, sr=16000):
    """Extracts chroma features from an audio file."""
    y, sr = librosa.load(file_path, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma.T, axis=0)

def extract_features(file_path):
    """Extracts multiple audio features and returns as a combined feature vector."""
    mfcc = extract_mfcc(file_path)
    chroma = extract_chroma(file_path)
    return np.concatenate([mfcc, chroma])

def process_dataset(input_dir, output_csv):
    """Processes all audio files in a directory and saves extracted features to a CSV file."""
    import os
    features_list = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_dir, file_name)
            features = extract_features(file_path)
            features_list.append(features)
    
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_directory = "../data/raw/"
    output_file = "../data/processed/audio_features.csv"
    
    print("Extracting features from dataset...")
    process_dataset(input_directory, output_file)
    print(f"Feature extraction completed. Saved to {output_file}")



# This script:

# Extracts MFCC features from an audio file.
# Extracts Chroma features for pitch and tonal analysis.
# Combines extracted features into a single feature vector.
# Processes an entire dataset and saves features in audio_features.csv.




# Variant 2

# import pandas as pd
# import numpy as np

# def load_features(csv_path):
    # df = pd.read_csv(csv_path)
    # X = df.drop(columns=['label']).values
    # y = df['label'].values
    # return X, y
