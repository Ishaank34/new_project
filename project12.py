import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Display an example image
plt.matshow(X_train[0])
plt.show()

# ANN Model
ann_model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')  # Use softmax for multi-class classification
])

ann_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train ANN Model
ann_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate ANN Model
ann_eval = ann_model.evaluate(X_test, y_test)
print(f"ANN Test Loss: {ann_eval[0]}, ANN Test Accuracy: {ann_eval[1]}")

# Reshape data for CNN input
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# CNN Model
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')  # Use softmax for multi-class classification
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train CNN Model
cnn_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate CNN Model
cnn_eval = cnn_model.evaluate(X_test, y_test)
print(f"CNN Test Loss: {cnn_eval[0]}, CNN Test Accuracy: {cnn_eval[1]}")