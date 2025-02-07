from src.models.cnn_model import CNNModel
from src.utils.feature_extraction import load_features
from sklearn.model_selection import train_test_split

# Load features
X, y = load_features('../data/processed/audio_features.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CNN
model = CNNModel(input_shape=X_train.shape[1:])
model.train(X_train, y_train)
model.save('../models/alzheimer_cnn.h5')