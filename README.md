# MNIST Classification using ANN and CNN

## Overview
This project implements classification of handwritten digits from the MNIST dataset using two different deep learning models:
1. *Artificial Neural Network (ANN)* - A simple feedforward neural network.
2. *Convolutional Neural Network (CNN)* - A deep learning model designed for image classification.

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9), each of size 28x28 pixels.

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

You can install the required libraries using:
sh
pip install tensorflow numpy matplotlib


## Code Explanation

### 1. Load and Preprocess the Data
- The dataset is loaded using keras.datasets.mnist.load_data().
- The pixel values are normalized to the range [0,1] by dividing by 255.

### 2. Build and Train ANN Model
- A sequential model with:
  - Flatten layer to convert 2D images into a 1D array.
  - A dense hidden layer with 100 neurons and ReLU activation.
  - A dense output layer with 10 neurons and softmax activation for classification.
- Compiled with Adam optimizer and sparse categorical cross-entropy loss.
- Trained for 10 epochs.

### 3. Build and Train CNN Model
- A sequential model with:
  - A Conv2D layer with 32 filters and a 3x3 kernel, using ReLU activation.
  - A MaxPooling2D layer to downsample the feature maps.
  - A Flatten layer to convert 2D features into 1D.
  - A dense hidden layer with 100 neurons and ReLU activation.
  - A dense output layer with 10 neurons and softmax activation.
- Compiled with Adam optimizer and sparse categorical cross-entropy loss.
- Trained for 5 epochs.

### 4. Evaluate Models
Both models are evaluated on the test dataset using model.evaluate().

## Results
The CNN model generally achieves higher accuracy than the ANN model due to its ability to extract spatial features.

## Running the Code
Run the script using:
sh
python mnist_classification.py


## Future Improvements
- Increase the number of filters in CNN.
- Add dropout layers to prevent overfitting.
- Experiment with different architectures (e.g., deeper networks, batch normalization).

## Author
Developed as part of a deep learning experiment for handwritten digit classification.

## License
This project is open-source and available for public use.
