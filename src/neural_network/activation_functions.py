from typing import Union

import numpy as np


class ActivationFunctions:
    """
    Activation Functions to be used in the neural network.

    Methods:
        sigmoid(z)
        relu(z)
        softmax(z)
        tanh(z)
    """

    @staticmethod
    def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the sigmoid activation function.

        The sigmoid function is defined as:

        σ(z) = 1 / (1 + e^{-z})

        Args:
            z (float or np.ndarray): Input value or array.

        Returns:
            float or np.ndarray: Sigmoid of the input.
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def relu(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the ReLU activation function.

        The ReLU function is defined as:

        ReLU(z) = max(0, z)

        Args:
            z (float or np.ndarray): Input value or array.

        Returns:
            float or np.ndarray: ReLU of the input.
        """
        return np.maximum(0, z)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax activation function.

        The softmax function is defined as:

        softmax(z_i) = e^{z_i} / Σ e^{z_j}

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Softmax of the input.
        """
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / exp_z.sum(axis=0, keepdims=True)

    @staticmethod
    def tanh(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the tanh activation function.

        The tanh function is defined as:

        tanh(z) = (e^{z} - e^{-z}) / (e^{z} + e^{-z})

        Args:
            z (float or np.ndarray): Input value or array.

        Returns:
            float or np.ndarray: Tanh of the input.
        """
        return np.tanh(z)
