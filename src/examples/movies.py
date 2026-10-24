"""Decides if one should go to the movies based on weather, company, proximity.

Reference: https://youtu.be/OFbnpY_k7js?t=133&si=trc9UtUHm8LKwb0e

Models the following:
Inputs:
   x1: Weather, 1 is good, 0 is bad.
   x2: Company, 1 is with company, 0 without.
   x3: Proximity, 1 is near, 0 is far.
Output:
   If weather is good and if ONE of the other factors is in favor = 1,
   else = 0
"""

import numpy as np

from ..core.perceptron import Perceptron
from ..core.functions import StepFunction

perceptron = Perceptron(inputs=3, w=np.array([4.0, 2.0, 2.0]), b=-4.0, f=StepFunction)

for weather in [0, 1]:
    for company in [0, 1]:
        for proximity in [0, 1]:
            print(f"Weather: {weather} | Company: {company} | Proximity: {proximity}")
            output = perceptron.a(np.array([weather, company, proximity], dtype=float))
            if output == 1:
                print("Will go to the movies :)\n")
            else:
                print("Won't go to the movies :(\n")
