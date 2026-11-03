import numpy as np
import math

from ...core.perceptron import Perceptron, FullyConnectedMLP
from ...core.activations import SigmoidFunction

input_layer = [
    Perceptron(1, SigmoidFunction, np.array([1.0]), 7.0),
]
hidden_layer_1 = [Perceptron(1, SigmoidFunction, np.array([1.0]), -6.0)]
output_layer = [Perceptron(1, SigmoidFunction, np.array([1.0]), -7.0)]

layers = [input_layer, hidden_layer_1, output_layer]

mlp = FullyConnectedMLP(1, layers)
X = np.array([1.0, 2.0, 5.0, -10.0, 5.0])
A = np.array([mlp.a([x]) for x in X])
Y = np.array([1.0, 0.5, 0.1, 0.0, 1.0])
print(f"X = {X}\nA = {A}\nY = {Y}")
print("Cost = ", mlp.cost(Y, A))

# FIX: the computation below is only correct for the last layer

# Computing all the partial derivatives of C with respect to (wrt) wL and bL
# where L is the current Layer.
# Which means that we will compute dC/dwL and dC/dbL for each layer.
next_inputs = X
for i, L in enumerate(reversed(layers)):
    inputs = next_inputs
    next_inputs = []
    dC_dwL = 0
    dC_dbL = 0
    # Computing the partial derivative of C wrt wL and C wrt bL
    # The cost is the average of all the costs for the inputs from the previous
    # layer (or the x inputs for the first one).
    # So we calculate all the partial derivatives and average them.

    for k, input_L in enumerate(inputs):
        # Compute the outputs from layer L - 1
        layer_index = 0
        while layer_index != i:
            inputs = mlp.a_for_layer(np.array([input_L]), layers[i])

        # Computing the partial derivative of C wrt wL, given the input x
        # According to the chain rule:
        # dCx_dwL = dzL_dwL * daL_dzL * dCx_daL
        # So, first, let's compute each partial derivative:
        a_L = mlp.a_for_layer(np.array([input_L]), L)[0]
        dzL_dwL = a_L
        daL_dzL = a_L * (1 - a_L)
        y = Y[i]
        dCx_daL = 2 * (a_L - y)

        dCx_dwL = dzL_dwL * daL_dzL * dCx_daL

        # Computing the partial derivative of C wrt bL, given x.
        dzL_dbL = 1
        dCx_dbL = dzL_dbL * daL_dzL * dCx_daL

        dC_dwL += dCx_dwL
        dC_dbL += dCx_dbL
        next_inputs.append(a_L)

    dC_dwL /= len(X)
    dC_dbL /= len(X)
