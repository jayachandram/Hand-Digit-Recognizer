# MNIST Neural Network from Scratch (NumPy Implementation)

## Project Overview
This project implements a simple neural network from scratch using only NumPy — without TensorFlow or Keras. It aims to classify handwritten digits from the MNIST dataset, which contains 70,000 images of digits (0-9) in 28x28 pixel grayscale format.

## Features
- Fully connected neural network architecture
- Forward and backward propagation implemented manually
- Cross-entropy loss function
- Stochastic Gradient Descent (SGD) optimisation
- Accuracy evaluation

## Project Structure
- `simple-mnist-nn-from-scratch-numpy-no-tf-keras.ipynb`: Jupyter notebook containing the complete implementation, including data loading, model definition, training, and evaluation.

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- NumPy
- Matplotlib
- Jupyter Notebook

Install the dependencies with:
```bash
pip install numpy matplotlib notebook
```

## How to Run
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook simple-mnist-nn-from-scratch-numpy-no-tf-keras.ipynb
   ```
2. Run the notebook cells sequentially.
3. The model will train and evaluate on the MNIST dataset.

## Model Architecture
- **Input layer:** 784 neurons (28x28 pixels)
- **Hidden layer:** 128 neurons (ReLU activation)
- **Output layer:** 10 neurons (Softmax activation)

## Training Details
- **Loss function:** Cross-Entropy
- **Optimiser:** Stochastic Gradient Descent (SGD)
- **Learning rate:** 0.01
- **Epochs:** 20
- **Batch size:** 32

## Results
The model achieves an accuracy of approximately ~92% on the test set after 20 epochs.

## Potential Improvements
- Implement additional hidden layers for deeper networks
- Add L2 regularisation or dropout to prevent overfitting
- Try advanced optimisation methods like Adam


---
Happy coding!

