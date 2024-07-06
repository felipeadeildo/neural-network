from typing import List, Literal

import numpy as np

from .cost_functions import CostFunctions
from .layers import DenseLayer


class NeuralNetwork:
    """
    Neural Network class to perform forward and backward propagation.

    Methods:
        forward(self, X)
        backward(self, Y_hat, Y)
        train(self, X, Y, epochs)
    """

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
        dA = -(np.divide(Y, Y_hat + 1e-8) - np.divide(1 - Y, 1 - Y_hat + 1e-8))

        for layer in reversed(self.layers):
            dA = layer.backward(dA, self.learning_rate)
            self.clip_gradients(dA)

    def clip_gradients(self, dA: np.ndarray, clip_value: float = 5.0) -> np.ndarray:
        """
        Clip gradients to avoid gradient explosion.

        Args:
            dA (np.ndarray): Gradients to be clipped.
            clip_value (float): Value to clip gradients.

        Returns:
            np.ndarray: Clipped gradients.
        """
        np.clip(dA, -clip_value, clip_value, out=dA)
        return dA

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int,
        cost_function: Literal["cross_entropy", "mean_squared_error"] = "cross_entropy",
        verbose: bool = True,
        verbose_freq: int = 10,
    ) -> None:
        """
        Train the neural network.

        Args:
            X (np.ndarray): Input data.
            Y (np.ndarray): True labels.
            epochs (int): Number of training epochs.
            cost_function (Literal["cross_entropy", "mean_squared_error"]): Cost function to use. Defaults to "cross_entropy".
            verbose (bool): Whether to print cost after every `verbose_freq` epochs.
            verbose_freq (int): Frequency of printing cost.
        """
        cost_fn = getattr(CostFunctions, cost_function)
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            self.backward(Y_hat, Y)
            if verbose and epoch % verbose_freq == 0:
                cost = cost_fn(Y_hat, Y)
                accuracy = self.compute_accuracy(Y_hat, Y)
                print(f"Cost after epoch {epoch}: {cost}, Accuracy: {accuracy:.2f}%")

    def compute_accuracy(self, Y_hat: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the accuracy of the model.

        Args:
            Y_hat (np.ndarray): Predicted outputs.
            Y (np.ndarray): True labels.

        Returns:
            float: Accuracy of the model.
        """
        predictions = np.argmax(Y_hat, axis=0)
        labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == labels) * 100
        return accuracy
