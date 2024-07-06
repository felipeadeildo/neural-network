from typing import Union

import numpy as np


class Derivatives:
    """
    Derivatives of Activation Functions to be used in the neural network.

    Methods:
        sigmoid_derivative(z)
        relu_derivative(z)
        tanh_derivative(z)
    """

    @staticmethod
    def sigmoid_derivative(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the derivative of the sigmoid function.

        The derivative of the sigmoid function is defined as:

        σ'(z) = σ(z) * (1 - σ(z))

        Args:
            z (float or np.ndarray): Input value or array.

        Returns:
            float or np.ndarray: Derivative of sigmoid of the input.
        """
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)

    @staticmethod
    def relu_derivative(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the derivative of the ReLU function.

        The derivative of the ReLU function is defined as:

        ReLU'(z) =
            1 if z > 0
            0 if z ≤ 0

        Args:
            z (float or np.ndarray): Input value or array.

        Returns:
            float or np.ndarray: Derivative of ReLU of the input.
        """
        return np.where(z > 0, 1, 0)

    @staticmethod
    def tanh_derivative(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the derivative of the tanh function.

        The derivative of the tanh function is defined as:

        tanh'(z) = 1 - tanh(z)^2

        Args:
            z (float or np.ndarray): Input value or array.

        Returns:
            float or np.ndarray: Derivative of tanh of the input.
        """
        return 1 - np.tanh(z) ** 2
