import librosa
import os

def preprocess_audio(file_path, output_path):
    y, sr = librosa.load(file_path, sr=16000)
    y_trimmed, _ = librosa.effects.trim(y)
    librosa.output.write_wav(output_path, y_trimmed, sr)

# Process all files
input_dir = '../data/raw/'
output_dir = '../data/processed/'
for file in os.listdir(input_dir):
    if file.endswith('.wav'):
        preprocess_audio(os.path.join(input_dir, file), os.path.join(output_dir, file))