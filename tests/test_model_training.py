# Model training

import numpy as np
import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.models.train_model import build_model, train_model

def test_build_model():
    """Test if the model is built correctly."""
    model = build_model(input_shape=13)
    
    assert isinstance(model, Sequential), "Model should be a Sequential model."
    assert len(model.layers) > 0, "Model should have at least one layer."

def test_train_model():
    """Test if the model training function runs without errors."""
    X_train = np.random.rand(100, 13)
    y_train = np.random.randint(0, 2, size=(100,))
    X_test = np.random.rand(20, 13)
    y_test = np.random.randint(0, 2, size=(20,))
    
    model = build_model(input_shape=13)
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=1, batch_size=10)
    
    assert history is not None, "Training history should not be None."

def test_train_model_invalid_data():
    """Test model training with invalid data input."""
    with pytest.raises(Exception):
        model = build_model(input_shape=13)
        train_model(model, None, None, None, None, epochs=1, batch_size=10)

if __name__ == "__main__":
    pytest.main()


# Test poprawności budowy modelu (czy model zawiera warstwy i jest typu Sequential).
# Test poprawności trenowania modelu (czy train_model zwraca historię treningu).
# Test obsługi błędów dla niepoprawnych danych wejściowych.