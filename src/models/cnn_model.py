
# Convolutional Neural Network (CNN)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_cnn_model(input_shape):
    """Build and compile a Convolutional Neural Network (CNN) model."""
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    """Example usage of CNN model."""
    input_shape = (13, 1)  # Example input shape for MFCC features
    model = build_cnn_model(input_shape)
    model.summary()

if __name__ == "__main__":
    main()


# This script:

# Builds a CNN model with:

# Conv1D layers for feature extraction.
# MaxPooling1D to reduce dimensions.
# Dropout layer to prevent overfitting.
# Sigmoid activation for binary classification.
# Compiles the model with:

# Adam optimizer.
# Binary cross-entropy loss.
# Accuracy as a performance metric.
# Includes an example usage (main() function) that prints the model architecture.





# Variant 2

# import tensorflow as tf

# class CNNModel:
    # def __init__(self, input_shape):
        # self.model = tf.keras.Sequential([
            # tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            # tf.keras.layers.MaxPooling1D(pool_size=2),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.Dense(1, activation='sigmoid')
        # ])
        # self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # def train(self, X_train, y_train):
        # self.model.fit(X_train, y_train, epochs=10, batch_size=32)

    # def save(self, model_path):
        # self.model.save(model_path)

    # @staticmethod
    # def load(model_path):
        # return tf.keras.models.load_model(model_path)


# Variant 3

#import torch.nn as nn

#class CNNModel(nn.Module):
    #def __init__(self, input_channels, num_classes):
        #super(CNNModel, self).__init__()
        #self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.fc = nn.Linear(32 * 32 * 32, num_classes)

    #def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = x.view(-1, 32 * 32 * 32)
        #return self.fc(x)