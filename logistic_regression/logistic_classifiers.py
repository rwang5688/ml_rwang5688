from math import exp

def sigmoid(x):
    return 1 / (1+exp(-x))

def f(x,p):
    return p + 0.35 * x - 0.56

def l(x,p):
    return sigmoid(f(x,p))

