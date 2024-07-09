from typing import Callable, Literal, Mapping, Union

import numpy as np


def sigmoid(
    z: Union[float, np.ndarray], derivative: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute the sigmoid activation function.

    The sigmoid function is defined as:
    σ(z) = 1 / (1 + e^{-z})

    The derivative of the sigmoid function is:
    σ'(z) = σ(z) * (1 - σ(z))

    Args:
        z (float or np.ndarray): Input value or array.
        derivative (bool, optional): Whether to compute the derivative of the activation function. Defaults to False.

    Returns:
        float or np.ndarray: Sigmoid of the input.
    """
    sig = 1 / (1 + np.exp(-z))
    if derivative:
        return sig * (1 - sig)
    return sig


def relu(
    z: Union[float, np.ndarray], derivative: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute the ReLU activation function.

    The ReLU function is defined as:
    ReLU(z) = max(0, z)

    The derivative of the ReLU function is:
    ReLU'(z) = 1 if z > 0, 0 otherwise

    Args:
        z (float or np.ndarray): Input value or array.
        derivative (bool, optional): Whether to compute the derivative of the activation function. Defaults to False.

    Returns:
        float or np.ndarray: ReLU of the input.
    """
    if derivative:
        return np.where(z > 0, 1, 0)
    return np.maximum(0, z)


def softmax(z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Compute the softmax activation function.

    The softmax function is defined as:
    softmax(z_i) = e^{z_i} / Σ e^{z_j}

    The derivative of the softmax function is not typically computed directly,
    but rather the gradients are computed during backpropagation.

    Args:
        z (np.ndarray): Input array.
        derivative (bool, optional): This parameter is ignored for softmax, included for compatibility.

    Returns:
        np.ndarray: Softmax of the input.
    """
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    softmax_output = exp_z / exp_z.sum(axis=0, keepdims=True)
    if derivative:
        return softmax_output * (1 - softmax_output)
    return softmax_output


def tanh(
    z: Union[float, np.ndarray], derivative: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute the tanh activation function.

    The tanh function is defined as:
    tanh(z) = (e^{z} - e^{-z}) / (e^{z} + e^{-z})

    The derivative of the tanh function is:
    tanh'(z) = 1 - tanh(z)^2

    Args:
        z (float or np.ndarray): Input value or array.
        derivative (bool, optional): Whether to compute the derivative of the activation function. Defaults to False.

    Returns:
        float or np.ndarray: Tanh of the input.
    """
    tanh_output = np.tanh(z)
    if derivative:
        return 1 - np.power(tanh_output, 2)
    return tanh_output


ACTIVATIONS: Mapping[Literal["sigmoid", "relu", "tanh", "softmax"], Callable] = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "softmax": softmax,
}
