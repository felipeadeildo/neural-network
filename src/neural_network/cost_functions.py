import numpy as np


class CostFunctions:
    """
    Cost Functions to be used in the neural network

    Methods:
        cross_entropy(y_hat, y)
        mean_squared_error(y_hat, y)
    """

    @staticmethod
    def cross_entropy(y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Cross-Entropy Loss function.

        The cross-entropy loss is defined as:

        L = - (1/n) Σ Σ y_{ij} log(y_hat_{ij})

        Args:
            y_hat (np.ndarray): Predicted probabilities.
            y (np.ndarray): Actual labels. (True Labels).

        Returns:
            float: Cross-Entropy Loss.
        """
        n = y.shape[1]  # number of examples
        epsilon = 1e-15  # small value to avoid log(0)
        cost = (-1 / n) * np.sum(
            y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon)
        )

        return float(np.squeeze(cost))  # squeeze removes dimensions of size 1

    @staticmethod
    def mean_squared_error(y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the mean squared error (MSE).

        The mean squared error is defined as:

        MSE = (1/n) Σ (y_i - y_hat_i)^2

        Args:
            y_hat (np.ndarray): Predicted values.
            y (np.ndarray): Actual values. (True Labels).

        Returns:
            float: Mean squared error.
        """
        return float(np.mean((y_hat - y) ** 2))
