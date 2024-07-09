from typing import Literal

import numpy as np

from .activations import ACTIVATIONS


class DenseLayer:
    """
    Dense Layer class to perform forward and backward propagation for a single dense layer.

    Methods:
        forward(self, A_prev)
        backward(self, dA, learning_rate)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Literal["sigmoid", "relu", "tanh", "softmax"],
    ) -> None:
        """
        Initialize a dense layer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of neurons in the layer.
            activation (str): Activation function to use ('sigmoid', 'relu', 'tanh', 'softmax').
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = ACTIVATIONS[activation]

        # Initialize weights and biases with He initialization for ReLU, normal for others
        if activation == "relu":
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(
                2.0 / input_size
            )
        else:
            self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the layer.

        Args:
            A_prev (np.ndarray): Activations from the previous layer.

        Returns:
            np.ndarray: Activations after applying the activation function.
        """
        self.input_data = input_data
        self.z = np.dot(self.weights, input_data) + self.biases
        self.a = self.activation(self.z)
        return self.a

    def backward(self, dA: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Perform backward propagation through the layer.

        Args:
            dA (np.ndarray): Gradient of the cost/loss with respect to the output.
            learning_rate (float): Learning rate for updating parameters.

        Returns:
            np.ndarray: Gradient of the cost/loss with respect to the activation of the previous layer. (input)
        """
        m = self.input_data.shape[1]
        dZ = dA * self.activation(self.z, derivative=True)
        dW = np.dot(dZ, self.input_data.T) / m
        dB = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.weights.T, dZ)

        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB

        return dA_prev
