import numpy as np
from collections.abc import Callable
import math
from .activations import ActivationFunctionType


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
        """Calculate the final activation for the whole network.

        Args:
            x: A list of the inputs
        """
        inputs = np.asarray(x, dtype=float)
        for layer in self.layers:
            next_layer_inputs = self.a_for_layer(inputs, layer)
            inputs = next_layer_inputs

        return inputs

    def a_for_layer(
        self, inputs: np.ndarray, layer: list[Perceptron] | int
    ) -> np.ndarray:
        if isinstance(layer, int):
            layer = self.layers[layer]

        return np.array([perceptron.a(inputs) for perceptron in layer], dtype=float)

    def cost_for_x(self, desired_a: np.ndarray, a: np.ndarray) -> float:
        """Calculates the squared error cost between desired and actual activation."""
        sub = desired_a - a
        magnitude = np.linalg.norm(sub)
        return math.pow(magnitude, 2)

    def cost(self, desired_a_arr: np.ndarray, a_arr: np.ndarray) -> float:
        """Calculates the mean squared error cost between desired and actual activations."""
        costs = np.empty(len(desired_a_arr))

        for i in range(len(desired_a_arr)):
            costs[i] = self.cost_for_x(desired_a_arr[i], a_arr[i])

        return costs.mean()

    def backpropagation_ai(
        self, x: np.ndarray, desired_a: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Calculate weight and bias gradients for one training example.

        Returns:
            A tuple containing weight gradients and bias gradients, both in
            the same layer order as ``self.layers``.
        """
        inputs = np.asarray(x, dtype=float).reshape(-1)
        desired = np.asarray(desired_a, dtype=float).reshape(-1)

        if inputs.size != self.inputs:
            raise ValueError(f"Expected {self.inputs} inputs, got {inputs.size}")

        activations = [inputs]
        pre_activations = []

        for layer in self.layers:
            layer_input = activations[-1]
            layer_z = np.array(
                [perceptron.z(layer_input) for perceptron in layer], dtype=float
            )
            layer_a = np.array(
                [perceptron.f(z) for perceptron, z in zip(layer, layer_z)],
                dtype=float,
            )
            pre_activations.append(layer_z)
            activations.append(layer_a)

        if activations[-1].size != desired.size:
            raise ValueError(
                f"Expected {activations[-1].size} desired outputs, got {desired.size}"
            )

        deltas = [None] * len(self.layers)
        output_layer = self.layers[-1]
        output_derivative = np.array(
            [
                self._activation_derivative(perceptron.f, z)
                for perceptron, z in zip(output_layer, pre_activations[-1])
            ],
            dtype=float,
        )
        deltas[-1] = 2 * (activations[-1] - desired) * output_derivative

        for layer_index in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]
            next_delta = deltas[layer_index + 1]
            propagated = np.array(
                [
                    sum(
                        next_delta[j] * next_layer[j].w[i]
                        for j in range(len(next_layer))
                    )
                    for i in range(len(layer))
                ],
                dtype=float,
            )
            derivative = np.array(
                [
                    self._activation_derivative(perceptron.f, z)
                    for perceptron, z in zip(layer, pre_activations[layer_index])
                ],
                dtype=float,
            )
            deltas[layer_index] = propagated * derivative

        weight_gradients = [
            delta[:, np.newaxis] * activations[layer_index][np.newaxis, :]
            for layer_index, delta in enumerate(deltas)
        ]
        bias_gradients = [delta.copy() for delta in deltas]

        return weight_gradients, bias_gradients

    @staticmethod
    def _activation_derivative(activation, z: float) -> float:
        epsilon = 1e-6
        return float(
            (activation(z + epsilon) - activation(z - epsilon)) / (2 * epsilon)
        )

    # TODO gradient descent

    # TODO backpropagation

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
