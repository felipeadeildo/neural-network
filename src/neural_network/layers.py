from typing import Literal

import numpy as np

from .activation_functions import ActivationFunctions
from .derivatives import Derivatives


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

        Raises:
            ValueError: If an unsupported activation function is specified.
        """
        # Initialize weights and biases
        if activation == "relu":
            self.W: np.ndarray = np.random.randn(output_size, input_size) * np.sqrt(
                2.0 / input_size
            )
        else:
            self.W: np.ndarray = np.random.randn(output_size, input_size) * 0.01
        self.b: np.ndarray = np.zeros((output_size, 1))

        self.activation_name = activation
        self.activation = getattr(ActivationFunctions, activation)
        self.activation_derivative = getattr(
            Derivatives, f"{activation}_derivative", None
        )

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the layer.

        Args:
            A_prev (np.ndarray): Activations from the previous layer.

        Returns:
            np.ndarray: Activations after applying the activation function.
        """
        self.A_prev: np.ndarray = A_prev
        self.Z: np.ndarray = np.dot(self.W, A_prev) + self.b
        self.A: np.ndarray = self.activation(self.Z)
        return self.A

    def backward(self, dA: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Perform backward propagation through the layer.

        Args:
            dA (np.ndarray): Gradient of the cost with respect to the activation.
            learning_rate (float): Learning rate for updating parameters.

        Returns:
            np.ndarray: Gradient of the cost with respect to the activation of the previous layer.
        """
        if self.activation_name != "softmax":
            dZ = dA * self.activation_derivative(self.Z)  # type: ignore
        else:
            dZ = dA  # Softmax handled separately in loss function

        dW = np.dot(dZ, self.A_prev.T) / self.A_prev.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / self.A_prev.shape[1]
        dA_prev = np.dot(self.W.T, dZ)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dA_prev
