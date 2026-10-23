from ..core.perceptron import Perceptron
from ..core.functions import StepFunction

# Models the following:
# Inputs:
#   x1: Weather, 1 is good, 0 is bad.
#   x2: Company, 1 is with company, 0 without.
#   x3: Proximity, 1 is near, 0 is far.
# Output:
#   If weather is good and if ONE of the other factors is in favor = 1,
#   else = 0

perceptron = Perceptron(inputs=3, w=[4, 2, 2], b=-4, a=StepFunction)

for weather in [0, 1]:
    for company in [0, 1]:
        for proximity in [0, 1]:
            print(f"Weather: {weather} | Company: {company} | Proximity: {proximity}")
            output = perceptron.z([weather, company, proximity])
            if output >= 1:
                print("Will go to the movies :)\n")
            else:
                print("Won't go to the movies :(\n")
