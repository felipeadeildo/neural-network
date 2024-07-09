from typing import Callable, List, Literal

import numpy as np

from .costs import COSTS
from .layers import DenseLayer


class NeuralNetwork:
    """
    Neural Network class to perform forward and backward propagation.

    Methods:
        forward(self, X)
        backward(self, y_pred, y, learning_rate, cost_function)
        train(self, X, y, epochs, learning_rate, cost_function, verbose_freq)
        predict(self, X)
    """

    def __init__(self) -> None:
        self.layers: List[DenseLayer] = []

    def add_layer(self, layer: DenseLayer) -> None:
        """
        Add a layer to the neural network.

        Args:
            layer (DenseLayer): Layer to be added.
        """
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the network.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(
        self,
        y_pred: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        cost_function: Callable,
    ) -> None:
        """
        Perform backward propagation through the network.

        Args:
            y_pred (np.ndarray): Predicted output.
            y (np.ndarray): True labels.
            learning_rate (float): Learning rate for gradient descent.
            cost_function (Callable): Cost function to use.
        """
        dA = cost_function(y, y_pred, derivative=True)

        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the accuracy of the predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy of the predictions.
        """
        predictions = np.argmax(y_pred, axis=0)
        labels = np.argmax(y_true, axis=0)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float = 0.01,
        cost_function: Literal["cross_entropy", "mean_squared_error"] = "cross_entropy",
        batch_size: int = 64,
        verbose_freq: int = 10,
    ) -> None:
        """
        Train the neural network.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
            epochs (int): Number of training epochs.
            learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            cost_function (Literal["cross_entropy", "mean_squared_error"]): Cost function to use. Defaults to "cross_entropy".
            batch_size (int, optional): Size of each batch for training. Defaults to 64.
            verbose_freq (int, optional): Frequency of verbose output. Defaults to 10.
        """
        cost_fn = COSTS[cost_function]
        m = X.shape[1]

        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i : i + batch_size]
                y_batch = y_shuffled[:, i : i + batch_size]

                y_pred = self.forward(X_batch)
                self.backward(y_pred, y_batch, learning_rate, cost_fn)

            if (epoch + 1) % verbose_freq == 0:
                y_pred = self.forward(X)
                cost = cost_fn(y, y_pred)
                accuracy = self.compute_accuracy(y, y_pred)
                print(f"Epoch {epoch}, Cost: {cost}, Accuracy: {accuracy:.2%}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the neural network.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted classes.
        """
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=0)
        return predictions
