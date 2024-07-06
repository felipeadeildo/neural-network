from typing import List

import numpy as np

from .cost_functions import CostFunctions
from .layers import DenseLayer


class NeuralNetwork:
    def __init__(self, layers: List[DenseLayer], learning_rate: float = 0.01) -> None:
        """
        Initialize the neural network.

        Args:
            layers (List[DenseLayer]): List of layers in the neural network.
            learning_rate (float): Learning rate for gradient descent.
        """
        self.layers = layers
        self.learning_rate = learning_rate

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the network.
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, Y_hat: np.ndarray, Y: np.ndarray) -> None:
        """
        Perform backward propagation through the network.

        Args:
            Y_hat (np.ndarray): Predicted outputs.
            Y (np.ndarray): True labels.
        """
        dA = -(np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

        for layer in reversed(self.layers):
            dA = layer.backward(dA, self.learning_rate)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int) -> None:
        """
        Train the neural network.

        Args:
            X (np.ndarray): Input data.
            Y (np.ndarray): True labels.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            self.backward(Y_hat, Y)
            if epoch % 100 == 0:
                cost = CostFunctions.cross_entropy(Y_hat, Y)
                print(f"Cost after epoch {epoch}: {cost}")
