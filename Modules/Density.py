import numpy as np
from Modules.Distributions import Gaussian1D
from Modules.Estimators import MLE, EM1D

def FitDensityEstimate1D(x, y, num_alphas, iterations, algorithm = "MLE"):
    mu = {}
    sigma = {}
    alphas = {}
    class_conditional = {}
    priors = {}
    classes = list(set(y))
    epsilon = 1e-6
    
    num_samples = x.shape[0]
    num_features = x.shape[1]

    if(algorithm == "MLE"):
        z = 0
        for i in classes:
            temp = (y == i)
            priors[i] = x[temp].shape[0] / num_samples
            x_ = x[temp]

            mu_, sigma_ = MLE(x_)
            alpha_ = np.ones([1, num_features])
            class_conditional[i] = Gaussian1D(x, mu_, sigma_)
            class_conditional[i] = np.prod(class_conditional[i], axis = 1, keepdims = True)

            mu[i] = mu_
            sigma[i] = sigma_
            alphas[i] = alpha_

            z += priors[i] * class_conditional[i]

        posteriors = class_conditional[classes[0]] * priors[classes[0]] / (z + epsilon)
        for i in range(1, len(classes)):
            i_ = classes[i]
            t = class_conditional[i_] * priors[i_] / (z + epsilon)
            posteriors = np.concatenate([posteriors, t], axis = 1)
    elif(algorithm == "EM"):
        z = 0
        for i in classes:
            temp = (y == i)
            priors[i] = x[temp].shape[0] / num_samples
            x_ = x[temp]
            
            alpha_, mu_, sigma_ = EM1D(x_, num_alphas[i], iterations = iterations)
            mu[i] = mu_
            sigma[i] = sigma_
            alphas[i] = alpha_

            x_ = x.T
            x_ = x_.reshape([1, num_features, num_samples])
            
            mu_ = mu_.reshape(mu_.shape[0], mu_.shape[1], 1)
            sigma_ = sigma_.reshape(sigma_.shape[0], sigma_.shape[1], 1)
            alpha_ = alpha_.reshape(alpha_.shape[0], alpha_.shape[1], 1)
            
            class_conditional[i] = (Gaussian1D(x_, mu_, sigma_) * alpha_).sum(axis = 0)
            class_conditional[i] = np.prod(class_conditional[i], axis = 0, keepdims = True)
            # class_conditional[i][np.isnan(class_conditional[i])] = 0
            class_conditional[i] = class_conditional[i].reshape(num_samples, 1)
            
            z += priors[i] * class_conditional[i]

        posteriors = class_conditional[classes[0]] * priors[classes[0]] / z
        for i in range(1, len(classes)):
            i_ = classes[i]
            t = class_conditional[i_] * priors[i_] / z
            posteriors = np.concatenate([posteriors, t], axis = 1)
        
    return priors, alphas, mu, sigma, posteriors

def PredictDensityEstimate1D(x, mu, sigma, classes, priors, num_alphas):
    class_conditional = {}
    
    num_samples = x.shape[0]
    num_features = x.shape[1]
    
    epsilon = 1e-6
    
    z = 0
    for i in classes:
        mu_ = mu[i]
        mu_ = mu_.reshape(mu_.shape[0], mu_.shape[1], 1)
        sigma_ = sigma[i]
        sigma_ = sigma_.reshape(sigma_.shape[0], sigma_.shape[1], 1)
        alpha_ = num_alphas[i]
        alpha_ = alpha_.reshape(alpha_.shape[0], alpha_.shape[1], 1)
        x_ = x.T
        x_ = x_.reshape(1, num_features, num_samples)

        class_conditional[i] = (alpha_ * Gaussian1D(x_, mu_, sigma_)).sum(axis = 0)
        class_conditional[i] = np.prod(class_conditional[i], axis = 0, keepdims = True)
        # class_conditional[i][np.isnan(class_conditional[i])] = 0
        class_conditional[i] = class_conditional[i].reshape(num_samples, 1)
        
        z += priors[i] * class_conditional[i]
    
    posteriors = class_conditional[classes[0]] * priors[classes[0]] / (z + epsilon)
    for i in range(1, len(classes)):
        i_ = classes[i]
        t = class_conditional[i_] * priors[i_] / (z + epsilon)
        posteriors = np.concatenate([posteriors, t], axis = 1)

    return posteriors