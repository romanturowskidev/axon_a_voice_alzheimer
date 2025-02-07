from flask import Flask, request, jsonify
from src.models.cnn_model import CNNModel

app = Flask(__name__)
model = CNNModel.load('../models/alzheimer_cnn.h5')

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    features = extract_features(audio_file)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)