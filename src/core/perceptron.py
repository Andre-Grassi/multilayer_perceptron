import numpy as np
from collections.abc import Callable
from .functions import ActivationFunctionType


class Perceptron:
    """A neuron.

    Attributes:
        inputs: An integer count of the number of inputs to the neuron.
        w: NumPy array of weights for each input. Length must be equal to inputs.
        b: The bias, as float.
        f: The activation function, must follow ActivationFunctionType

    """

    def __init__(
        self, inputs: int, w: np.ndarray, b: float, f: ActivationFunctionType
    ):
        self.inputs = inputs
        self.w = np.asarray(w, dtype=float)
        self.b = b
        self.f = f

        if inputs != len(self.w):
            raise ValueError("Number of inputs different than number of weights")

    def z(self, x: np.ndarray) -> float:
        """z = w.x + b"""
        x_arr = np.asarray(x, dtype=float)
        return float(np.dot(self.w, x_arr) + self.b)

    def a(self, x: np.ndarray) -> float:
        """a = f(z)"""
        return self.f(self.z(x))


class FullyConnectedMLP:
    """A fully connected multi-layer perceptron network.

    Attributes:
        inputs: An integer count of the number of inputs to the neuron.
        layers: List of Lists of Perceptrons. Each List is a Layer, starting
        from the hidden layer all the way to the output layer.
    """

    def __init__(self, inputs: int, layers: list[list[Perceptron]]):
        self.inputs = inputs
        self.layers = layers

    def a(self, x: np.ndarray) -> np.ndarray:
        inputs = np.asarray(x, dtype=float)
        for layer in self.layers:
            next_layer_inputs = np.array(
                [perceptron.a(inputs) for perceptron in layer], dtype=float
            )
            inputs = next_layer_inputs

        outputs = next_layer_inputs

        return outputs
