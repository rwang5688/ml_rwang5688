import numpy as np
from math import exp


def sigmoid(x):
    return 1 / (1+exp(-x))


class RandomMLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [
            np.random.rand(n, m)
            for m,n in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.random.rand(n) for n in layer_sizes[1:]]

    def feedforward(self,v):
        activations = []
        a = v
        activations.append(a)
        for w,b in zip(self.weights, self.biases):
            z = w @ a + b
            a = [sigmoid(x) for x in z]
            activations.append(a)
        return activations

    def evaluate(self,v):
        return np.array(self.feedforward(v)[-1])

