import numpy as np
import math

from ..core.perceptron import Perceptron, FullyConnectedMLP
from ..core.activations import SigmoidFunction

input_layer = [
    Perceptron(1, SigmoidFunction, np.array([1.0]), 7.0),
]
hidden_layer_1 = [Perceptron(1, SigmoidFunction, np.array([1.0]), -6.0)]
hidden_layer_2 = [Perceptron(1, SigmoidFunction, np.array([1.0]), 6.0)]
output_layer = [Perceptron(1, SigmoidFunction, np.array([1.0]), -7.0)]

layers = [input_layer, hidden_layer_1, hidden_layer_2, output_layer]

mlp = FullyConnectedMLP(1, layers)
x = 1
a = mlp.a(np.array([x]))[0]
y = 1.0
print(f"x = {x} | a = {a} | desired = {y}")
print("Cost = ", mlp.cost_for_x(np.array(y), np.array(a)))

# Computing all the partial derivatives of C with respect to (wrt) w_L
# where L is the current Layer.

inputs = np.array(x)
for L in layers:
    # According to the chain rule:
    # dC_dw = dz_dw * da_dz * dC_da
    # So, first, let's compute each partial derivative:
    a_L = mlp.a_for_layer(inputs, L)[0]
    dz_dw = a_L
    da_dz = a_L * (1 - a_L)
    dC_da = 2 * (a_L - y)

    dC_dw = dz_dw * da_dz * dC_da

    inputs = np.array(a)
