import numpy as np

def Distance(x1, x2, metric = "Euclidean", axis = 0):
    if(metric == "Euclidean"):
        return np.linalg.norm(x1 - x2, axis = axis)
    elif(metric == "Manhattan"):
        return (np.abs(x1 - x2)).sum(axis = axis)
    elif(metric == "Chebyshev"):
        return np.max(np.abs(x1 - x2), axis = axis)
    elif(metric == "Minkowski"):
        n = 3
        return ((np.abs(x1 - x2) ** n).sum(axis = axis)) ** (1 / n)
    elif(metric == "Hamming"):
        return (x1 != x2).sum(axis = axis)