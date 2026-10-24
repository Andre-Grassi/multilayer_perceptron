"""Models a complex linear classifier, with 4 linear functions.

Reference: https://youtu.be/torNuKNLwBE?si=9byhUvjVEl3VP5UN&t=73

There is an error on the video: the fourth function is written as
5x_1 - 2x_2 - 20 = 0
But actually it is meant to be
-5x_1 + 2x_2 + 20 = 0.

This error was corrected in this code.
"""

from ..core.perceptron import Perceptron, FullyConnectedMLP
from ..core.functions import StepFunction

hidden_layer = [
    Perceptron(2, [-1, -1], 7, StepFunction),
    Perceptron(2, [2, 2], -6, StepFunction),
    Perceptron(2, [2, -2], 6, StepFunction),
    Perceptron(2, [-5, 2], 20, StepFunction),
]

output_layer = [Perceptron(4, [2, 2, 2, 2], -7, StepFunction)]

mlp = FullyConnectedMLP(2, [hidden_layer, output_layer])
print("x1 = 0 | x2 = 3 :", mlp.a([0, 0]))
print("x1 = 2 | x2 = 4 :", mlp.a([2, 4]))
