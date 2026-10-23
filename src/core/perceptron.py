from collections.abc import Callable
from .functions import ActivationFunctionType


class Perceptron:
    def __init__(
        self, inputs: int, w: list[float], b: float, a: ActivationFunctionType
    ):
        self.inputs = inputs
        self.w = w
        self.b = b
        self.a = a

        if inputs != len(w):
            raise ValueError("Number of inputs different than number of weights")

    def z(self, x: list[float]):
        z = 0
        for i in range(self.inputs):
            z += x[i] * self.w[i]
        z += self.b
        return z
