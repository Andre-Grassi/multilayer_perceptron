from typing import Protocol
import math


class ActivationFunctionType(Protocol):
    def __call__(self, z: float) -> float: ...


def StepFunction(z: float) -> float:
    if z <= 0:
        return 0
    return 1


def SigmoidFunction(z: float) -> float:
    return 1 / (1 + math.exp(-z))
