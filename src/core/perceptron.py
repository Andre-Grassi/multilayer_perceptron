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
        self,
        inputs: int,
        f: ActivationFunctionType,
        w: np.ndarray | None = None,
        b: float = 0,
    ):
        self.inputs = inputs
        self.f = f

        if w is not None:
            self.w = w
        else:
            rng = np.random.default_rng()
            self.w = rng.random(size=(inputs))

        self.b = b

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

        self._check_layer_dimensions(inputs, layers)

    def a(self, x: np.ndarray) -> np.ndarray:
        inputs = np.asarray(x, dtype=float)
        for layer in self.layers:
            next_layer_inputs = np.array(
                [perceptron.a(inputs) for perceptron in layer], dtype=float
            )
            inputs = next_layer_inputs

    def _check_layer_dimensions(self, inputs, layers):
        prev_dim = inputs
        for layer_idx, layer in enumerate(layers):
            for p_idx, perceptron in enumerate(layer):
                if perceptron.inputs != prev_dim:
                    raise ValueError(
                        f"Layer {layer_idx} perceptron {p_idx} expects {perceptron.inputs} inputs, "
                        f"but previous layer output dimension is {prev_dim}."
                    )
            prev_dim = len(layer)
