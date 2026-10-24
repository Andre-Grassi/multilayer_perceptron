from collections.abc import Callable
from .functions import ActivationFunctionType


class Perceptron:
    """A neuron.

    Attributes:
        inputs: An integer count of the number of inputs to the neuron.
        w: List of weights for each input. Length must be equal to inputs.
        b: The bias, as float.
        f: The activation function, must follow ActivationFunctionType

    """

    def __init__(
        self, inputs: int, w: list[float], b: float, f: ActivationFunctionType
    ):
        self.inputs = inputs
        self.w = w
        self.b = b
        self.f = f

        if inputs != len(w):
            raise ValueError("Number of inputs different than number of weights")

    def z(self, x: list[float]) -> float:
        """z = w.x + b"""

        z = 0
        for i in range(self.inputs):
            z += x[i] * self.w[i]
        z += self.b
        return z

    def a(self, x: list[float]) -> float:
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

    def a(self, x: list[float]) -> list[float]:
        inputs = x
        outputs = []
        next_layer_inputs = []
        for layer in self.layers:
            next_layer_inputs = []

            for perceptron in layer:
                next_layer_inputs.append(perceptron.a(inputs))

            inputs = next_layer_inputs

        outputs = next_layer_inputs

        return outputs
