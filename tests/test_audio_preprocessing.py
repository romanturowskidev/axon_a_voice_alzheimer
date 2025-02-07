from src.utils.audio_preprocessing import preprocess_audio

def test_audio_preprocessing():
    input_file = '../data/raw/sample.wav'
    output_file = '../data/processed/sample.wav'
    preprocess_audio(input_file, output_file)
    assert os.path.exists(output_file)
