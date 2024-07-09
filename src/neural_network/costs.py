from typing import Callable, Literal, Mapping, Union

import numpy as np


def cross_entropy(
    y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute the Cross-Entropy Loss function.

    The cross-entropy loss is defined as:
    L = - (1/n) Σ Σ y_{ij} log(y_hat_{ij})

    The derivative of the cross-entropy loss is:
    L = - (1/n) Σ y_{ij} (1 - y_hat_{ij})

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.
        derivative (bool, optional): Whether to return the derivative of the loss function.

    Returns:
        float or np.ndarray: Cross-Entropy Loss or its derivative.
    """
    epsilon = 1e-10  # Small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = y_true.shape[1]
    if derivative:
        return (y_pred - y_true) / m
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m


def mean_squared_error(
    y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute the mean squared error (MSE).

    The mean squared error is defined as:
    MSE = (1/n) Σ (y_i - y_hat_i)^2

    The derivative of the mean squared error is:
    MSE = (1/n) Σ 2(y_i - y_hat_i)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.
        derivative (bool, optional): Whether to return the derivative of the loss function.

    Returns:
        float or np.ndarray: Mean squared error or its derivative.
    """
    if derivative:
        return y_pred - y_true
    return np.mean(np.power(y_true - y_pred, 2))  # type: ignore


COSTS: Mapping[Literal["cross_entropy", "mean_squared_error"], Callable] = {
    "cross_entropy": cross_entropy,
    "mean_squared_error": mean_squared_error,
}
