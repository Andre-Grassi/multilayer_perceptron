"""Models a complex linear classifier, with 4 linear functions.

Reference: https://youtu.be/torNuKNLwBE?si=9byhUvjVEl3VP5UN&t=73

There is an error on the video: the fourth function is written as
5x_1 - 2x_2 - 20 = 0
But actually it is meant to be
-5x_1 + 2x_2 + 20 = 0.

This error was corrected in this code.
"""

import numpy as np

from ..core.perceptron import Perceptron, FullyConnectedMLP
from ..core.activations import StepFunction

hidden_layer = [
    Perceptron(2, StepFunction, np.array([-1.0, -1.0]), 7.0),
    Perceptron(2, StepFunction, np.array([2.0, 2.0]), -6.0),
    Perceptron(2, StepFunction, np.array([2.0, -2.0]), 6.0),
    Perceptron(2, StepFunction, np.array([-5.0, 2.0]), 20.0),
]

output_layer = [Perceptron(4, StepFunction, np.array([2.0, 2.0, 2.0, 2.0]), -7.0)]

mlp = FullyConnectedMLP(2, [hidden_layer, output_layer])
print("x1 = 0 | x2 = 0 :", mlp.a(np.array([0.0, 0.0])))
print("x1 = 2 | x2 = 4 :", mlp.a(np.array([2.0, 4.0])))
