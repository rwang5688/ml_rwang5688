from typing import List, Callable
from vectors import dot_product


class Neuron:
    def __init__(self,
        weights: List[float],
        learning_rate: float,
        activation_function: Callable[[float], float],
        derivative_activation_function: Callable[[float], float]) -> None:
        self.weights: List[float] = weights
        self.activation_function: Callable[[float], float] = activation_function
        self.derivative_activation_function: Callable[[float], float] = derivative_activation_function
        self.learning_rate: float = learning_rate
        self.weighted_input: float = 0.0
        self.delta: float = 0.0

    def output(self, inputs: List[float]) -> float:
        self.weighted_input = dot_product(inputs, self.weights)
        return self.activation_function(self.weighted_input)

