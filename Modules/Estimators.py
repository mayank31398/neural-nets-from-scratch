import numpy as np
from Modules.Distributions import Gaussian1D

def MLE(x):
    n = x.shape[0]
    mu = x.sum(axis = 0, keepdims = True) / n
    sigma = np.sqrt((np.square(x - mu)).sum(axis = 0, keepdims = True) / n)
    return mu, sigma

def EM1D(x, num_alpha, iterations):
    num_features = x.shape[1]
    num_samples = x.shape[0]
    n = max(num_alpha)
    
    mu = np.random.uniform(1, 2, size = [n, num_features])
    sigma = np.random.uniform(1, 2, size = [n, num_features])
    alpha = np.zeros([n, num_features])
    
    for i in range(len(num_alpha)):
        if(num_alpha[i] == 1):
            alpha[0, i] = 1
            mu[0, i], sigma[0, i] = MLE(x[:, i])
        else:
            x_ = x[:, i].reshape(num_samples, 1)
            alpha_ = np.zeros([1, num_alpha[i]]) + 1 / num_alpha[i]
            mu_ = np.random.uniform(1, 2, size = [1, num_alpha[i]])
            sigma_ = np.random.uniform(1, 2, size = [1, num_alpha[i]])
            
            if(num_alpha[i] < n):
                mu[num_alpha[i] + 1:, i] = 0
            
            for j in range(iterations):
                gaussians_ = Gaussian1D(x_, mu_, sigma_)
                w = gaussians_ * alpha_
                w /= w.sum(axis = 1, keepdims = True)
                w_sum = w.sum(axis = 0, keepdims = True)

                mu_ = (w * x_).sum(axis = 0, keepdims = True) / w_sum

                sigma_ = np.sqrt((w * (x_ - mu_) ** 2).sum(axis = 0, keepdims = True) / w_sum)

                alpha_ = w_sum / num_samples

            alpha[0:num_alpha[i], i] = alpha_
            mu[0:num_alpha[i], i] = mu_.reshape(num_alpha[i])
            sigma[0:num_alpha[i], i] = sigma_.reshape(num_alpha[i])
    
    return alpha, mu, sigma

def EMdD(x, num_alpha, algorithm, iterations):
    num_samples = x.shape[0]
    num_features = x.shape[1]
    
    if(algorithm == "MLE"):
        alphas = np.ones([1, 1])
        mu = x.sum(axis = 0, keepdims = True) / num_samples
        x_ = x - mu
        x_ = x_.reshape(num_samples, num_features, 1)
        x_ = np.matmul(x_, x_.transpose([0, 2, 1]))
        sigma = x_.sum(axis = 0) / num_samples
    elif(algorithm == "EM"):
        mu = np.random.uniform(1, 2, size = [num_alpha, num_features])
        
        sigma = np.abs(np.random.randn(num_alpha, num_features, num_features))
        sigma = 0.5 * (sigma + sigma.transpose([0, 2, 1]))
        sigma += num_features * np.eye(num_features).reshape(1, num_features, num_features)
        
        alphas = np.zeros([1, num_alpha]) + 1 / num_alpha

        for i in range(iterations):
            mu_ = mu.reshape(num_alpha, num_features, 1)
            x_ = x.T
            x_ = x_.reshape(1, num_features, num_samples)

            gaussians_ = x_ - mu_
            gaussians_ = gaussians_.transpose([0, 2, 1])
            gaussians_ = np.matmul(gaussians_, np.linalg.inv(sigma))
            gaussians_ = (gaussians_ * (x_ - mu_).transpose([0, 2, 1])).sum(axis = 2)
            gaussians_ = np.exp(-gaussians_ / 2)
            gaussians_ /= (np.sqrt(np.linalg.det(sigma).reshape(num_alpha, 1)) * (2 * np.pi) ** (num_features / 2))
            gaussians_ = gaussians_.T
            gaussians_ *= alphas

            w = gaussians_ / gaussians_.sum(axis = 1, keepdims = True)
            w_sum = w.sum(axis = 0, keepdims = True)

            w_ = w.T
            w_ = w_.reshape(num_alpha, num_samples, 1)
            x_ = x.reshape(1, num_samples, num_features)
            mu = (w_ * x_).sum(axis = 1) / (w_sum.T + 1e-6)

            x_ = x_.reshape(1, num_samples, num_features)
            mu_ = mu.reshape(num_alpha, 1, num_features)
            x_ = (x_ - mu_).reshape(num_alpha, num_samples, num_features, 1)
            x_ = np.matmul(x_, x_.transpose([0, 1, 3, 2]))
            sigma = (w_.T.reshape(num_alpha, num_samples, 1, 1) * x_).sum(axis = 1) / (w_sum.T.reshape(num_alpha, 1, 1) + 1e-6)

            alphas = w_sum / num_samples

    return alphas, mu, sigma