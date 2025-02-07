
# Recurrent Neural Network (RNN)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout

def build_rnn_model(input_shape):
    """Build and compile a Recurrent Neural Network (RNN) model using LSTM."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    """Example usage of RNN model."""
    input_shape = (13, 1)  # Example input shape for MFCC features
    model = build_rnn_model(input_shape)
    model.summary()

if __name__ == "__main__":
    main()


# This script:

# Builds an RNN model with:

# Two LSTM layers (64 and 128 units) to capture temporal dependencies.
# Dense layer for final classification.
# Dropout layer to prevent overfitting.
# Sigmoid activation for binary classification.
# Compiles the model with:

# Adam optimizer.
# Binary cross-entropy loss.
# Accuracy as a performance metric.
# Includes an example usage (main() function) that prints the model architecture.