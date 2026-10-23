from typing import Protocol


class ActivationFunctionType(Protocol):
    def __call__(self, z: float) -> float: ...


def StepFunction(z: float) -> float:
    if z <= 0:
        return 0
    return 1
