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
