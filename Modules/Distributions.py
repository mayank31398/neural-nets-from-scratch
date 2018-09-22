import numpy as np

def Gaussian1D(x, mu, sigma):
    epsilon = 1e-6
    return (1 / ((sigma + epsilon) * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2 + epsilon))

def GaussiandD(x, mu, sigma):
    d = x.shape[0]
    return (1 / (np.sqrt(np.linalg.det(sigma) * (2 * np.pi) ** d))) * np.exp(-np.matmul(np.matmul((x - mu).T, np.linalg.inv(sigma)), x - mu) / 2)