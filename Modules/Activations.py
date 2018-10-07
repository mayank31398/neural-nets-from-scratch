import numpy as np

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Tanh(x):
    return 2 * Sigmoid(x) - 1

def ReLU(x):
    return np.max(0, x)

def Softmax(x, axis = 0):
    t = np.exp(x)
    return t / t.sum(axis = axis, keepdims = True)