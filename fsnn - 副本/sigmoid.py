from math import exp


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def derivative_sigmoid(x: float) -> float:
    sig: float = sigmoid(x)
    return sig * (1 - sig)

