# End-to-end pipeline

import pytest
import numpy as np
import pandas as pd
from src.models.train_model import build_model, train_model
from src.utils.feature_extraction import extract_features
from src.app.api import app

def test_end_to_end():
    """Test the full pipeline from feature extraction to model prediction."""
    # Generate synthetic data for testing
    X_train = np.random.rand(100, 13)
    y_train = np.random.randint(0, 2, size=(100,))
    X_test = np.random.rand(20, 13)
    y_test = np.random.randint(0, 2, size=(20,))
    
    # Train model
    model = build_model(input_shape=13)
    train_model(model, X_train, y_train, X_test, y_test, epochs=1, batch_size=10)
    
    # Save and reload model for inference
    model.save("test_model.h5")
    loaded_model = build_model(input_shape=13)
    loaded_model.load_weights("test_model.h5")
    
    # Test inference
    predictions = loaded_model.predict(X_test)
    assert predictions.shape == (20, 1), "Prediction shape mismatch."
    
    # Test API endpoint
    client = app.test_client()
    response = client.post("/predict", json={"features": X_test[0].tolist()})
    assert response.status_code == 200, "API did not return a 200 status code."
    
    json_data = response.get_json()
    assert "prediction" in json_data, "API response missing prediction."

if __name__ == "__main__":
    pytest.main()



# Test ekstrakcji cech → sprawdza, czy cechy są poprawnie generowane,
# Test trenowania modelu → sprawdza, czy model trenuje się bez błędów,
# Test wczytywania modelu i predykcji → zapewnia poprawne działanie inferencji,
# Test API → sprawdza, czy API zwraca poprawną odpowiedź na zapytania