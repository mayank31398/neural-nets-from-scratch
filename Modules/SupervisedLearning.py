import numpy as np
import matplotlib.pyplot as plt
from Modules.Density import FitDensityEstimate1D, PredictDensityEstimate1D
from Modules.Estimators import EMdD
from Modules.Metrics import Accuracy, roc_curve
from Modules.Distances import Distance
from Modules.Activations import Sigmoid, Softmax

class NaiveBayesClassifier:
    training_accuracy = 0
    mu = None
    sigma = None
    classes = None
    priors = None
    alphas = None
    algorithm = None
    
    def __init__(self, algorithm = "MLE"):
        self.algorithm = algorithm
    
    def Fit(self, x, y, num_alphas, iterations = 100, roc_plot = False, label = None):
        """
        x = x_train, y = y_train\n
        iterations = 100, roc_plot = False, num_alphas = number of Gaussians to be used for a given class for a given feature 
        (dictionary of class to array of natural numbers with array size = num_features)\n
        label = class which you want roc for (default is None). If label == None, all classes' ROC is plotted.\n
        Returns predictions.
        """
        classes = list(set(y))
        
        num_samples = x.shape[0]

        self.priors, self.alphas, self.mu, self.sigma, posteriors = FitDensityEstimate1D(x, y, num_alphas, iterations, self.algorithm)

        temp = np.argmax(posteriors, axis = 1).reshape(num_samples, 1)
        classes = np.array(classes)
        
        if(roc_plot):
            ax = plt.subplot(1, 1, 1)

            if(label == None):
                for i in classes:
                    tpr, fpr, _ = roc_curve(y, np.max(posteriors, axis = 1), pos_label=i)
                    ax.plot(fpr, tpr, label = i)
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels)
            else:
                tpr, fpr, _ = roc_curve(y, np.max(posteriors, axis = 1), pos_label=label)
                ax.plot(fpr, tpr)
            
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("ROC curve")


        predictions = classes[temp]
        self.training_accuracy = Accuracy(predictions, y)
        self.classes = classes

        return predictions
    
    def Predict(self, x):
        """
        x = x_test\n
        Returns predictions.
        """
        classes = self.classes
        
        num_samples = x.shape[0]
        
        posteriors = PredictDensityEstimate1D(x, self.mu, self.sigma, self.classes, self.priors, self.alphas)
        
        temp = np.argmax(posteriors, axis = 1).reshape(num_samples, 1)
        classes = np.array(classes)

        predictions = classes[temp]
        
        return predictions

class BayesClassifier:
    training_accuracy = 0
    mu = None
    sigma = None
    classes = None
    priors = None
    alphas = None
    algorithm = None
    
    def __init__(self, algorithm = "MLE"):
        self.algorithm = algorithm
    
    def Fit(self, x, y, num_alphas, iterations = 100):
        """
        x = x_train, y = y_train\n
        iterations = 100, num_alphas = number of Gaussians to be used for a given class (dictionary of class to natural numbers.)\n
        label = class which you want roc for (default is None). If label == None, all classes' ROC is plotted.\n
        Returns predictions.
        """
        epsilon = 1e-10
        
        classes = list(set(y))
        self.mu = {}
        self.sigma = {}
        self.alphas = {}
        priors = {}
        class_conditional = {}
        
        num_features = x.shape[1]
        num_samples = x.shape[0]

        z = 0
        for i in classes:
            temp = (y == i)
            priors[i] = x[temp].shape[0] / num_samples
            x_ = x[temp]
            
            alpha_, mu_, sigma_ = EMdD(x_, num_alphas[i], self.algorithm, iterations = iterations)
            self.mu[i] = mu_
            self.sigma[i] = sigma_
            self.alphas[i] = alpha_
            
            mu_ = mu_.reshape(num_alphas[i], num_features, 1)
            x_ = x.T
            x_ = x_.reshape(1, num_features, num_samples)

            gaussians_ = x_ - mu_
            gaussians_ = gaussians_.transpose([0, 2, 1])
            gaussians_ = np.matmul(gaussians_, np.linalg.inv(sigma_))
            gaussians_ = np.exp(-(gaussians_ * (x_ - mu_).transpose([0, 2, 1])).sum(axis = 2) / 2)
            gaussians_ /= (np.sqrt(np.linalg.det(sigma_).reshape(num_alphas[i], 1) + epsilon) * (2 * np.pi) ** (num_features / 2))
            gaussians_ = gaussians_.T
            
            gaussians_ *= alpha_
            gaussians_ = gaussians_.sum(axis = 1, keepdims = True)
            
            class_conditional[i] = gaussians_
            z += gaussians_ * priors[i]
        
        posteriors = class_conditional[classes[0]] * priors[classes[0]] / (z + epsilon)
        for i in range(1, len(classes)):
            i_ = classes[i]
            t = class_conditional[i_] * priors[i_] / (z + epsilon)
            posteriors = np.concatenate([posteriors, t], axis = 1)
            
        temp = np.argmax(posteriors, axis = 1).reshape(num_samples, 1)
        classes = np.array(classes)

        predictions = classes[temp]
        
        self.training_accuracy = Accuracy(predictions, y.reshape(num_samples, 1))
        self.classes = classes
        self.priors = priors
        
        return predictions
    
    def Predict(self, x):
        """
        x = x_test\n
        Returns predictions.
        """
        class_conditional = {}
        classes = self.classes
        priors = self.priors
        epsilon = 1e-10
        
        num_features = x.shape[1]
        num_samples = x.shape[0]

        z = 0
        for i in classes:
            mu_ = self.mu[i]
            sigma_ = self.sigma[i]
            alpha_ = self.alphas[i]
            num_alphas = alpha_.shape[1]
            
            mu_ = mu_.reshape(num_alphas, num_features, 1)
            x_ = x.T
            x_ = x_.reshape(1, num_features, num_samples)

            gaussians_ = x_ - mu_
            gaussians_ = gaussians_.transpose([0, 2, 1])
            gaussians_ = np.matmul(gaussians_, np.linalg.inv(sigma_))
            gaussians_ = np.exp(-(gaussians_ * (x_ - mu_).transpose([0, 2, 1])).sum(axis = 2) / 2)
            gaussians_ /= (np.sqrt(np.linalg.det(sigma_).reshape(num_alphas, 1) + epsilon) * (2 * np.pi) ** (num_features / 2))
            gaussians_ = gaussians_.T
            
            gaussians_ *= alpha_
            gaussians_ = gaussians_.sum(axis = 1, keepdims = True)
            
            class_conditional[i] = gaussians_
            z += gaussians_ * priors[i]
        
        posteriors = class_conditional[classes[0]] * priors[classes[0]] / (z + epsilon)
        for i in range(1, len(classes)):
            i_ = classes[i]
            t = class_conditional[i_] * priors[i_] / (z + epsilon)
            posteriors = np.concatenate([posteriors, t], axis = 1)
            
        temp = np.argmax(posteriors, axis = 1).reshape(num_samples, 1)
        classes = np.array(classes)

        predictions = classes[temp]
        
        return predictions

class kNNClassifier:
    distance_metric = None

    def __init__(self, distance_metric = "Euclidean"):
        self.distance_metric = distance_metric
    
    def Predict(self, x, y, x_test, k):
        num_samples = x.shape[0]
        num_features = x.shape[1]
        num_test_samples = x_test.shape[0]
            
        classes = list(set(y))
        
        try:
            x_ = x.T
            x_ = x_.reshape(1, num_features, num_samples)
            x_test_ = x_test.reshape(num_test_samples, num_features, 1)
            
            dist = Distance(x_, x_test_, metric = self.distance_metric, axis = 1)
            temp = np.argpartition(dist, k - 1, axis = 1)[:, :k].reshape(num_test_samples, k)
            
            temp = y[temp]
            predictions = []
            for i in range(num_test_samples):
                count = {}
                
                for j in classes:
                    count[j] = 0
                
                for j in range(k):
                    count[temp[i, j]] += 1
                
                prediction = classes[0]
                for j in range(1, len(classes)):
                    if(count[classes[j]] > count[prediction]):
                        prediction = classes[j]
                predictions.append(prediction)
            
            predictions = np.array(predictions)
            
            return predictions
        except:
            predictions = []

            for j in range(num_test_samples):
                nearest = []
                for i in range(num_samples):
                    nearest.append([Distance(x[i, :], x_test[j, :], metric = self.distance_metric), y[i]])
                
                nearest.sort(key = lambda x: x[0])
                
                count = {}
                for i in classes:
                    count[i] = 0
                
                for i in range(k):
                    count[nearest[i][1]] += 1
                
                prediction = classes[0]
                for i in range(1, len(classes)):
                    if(count[classes[i]] > count[prediction]):
                        prediction = classes[i]

                predictions.append(prediction)
            predictions = np.array(predictions)
            
            return predictions

class ParzenWindow:
    h = None
    x_train = None
    y_train = None
    classes = None
    kernel = None

    def __init__(self, h = 1, kernel = "Hypercube"):
        self.h = h
        self.kernel = kernel
    
    def Fit(self, x, y):
        self.x_train = x
        self.y_train = y
        self.classes = list(set(y))
    
    def Predict(self, x):
        if(self.kernel == "Hypercube"):
            num_test_samples = x.shape[0]
            classes = self.classes
            h = self.h
            
            predictions = []
            for i in range(num_test_samples):
                x_ = self.x_train - x[i, :]
                x_ = np.abs(x_).max(axis = 1)
                temp_ = x_ < h / 2
                
                temp_ = self.y_train[temp_]
                count = 0
                prediction = classes[0]
                for j in classes:
                    t_ = (temp_ == j).sum()
                    if(t_ > count):
                        count = t_
                        prediction = j
                
                predictions.append(prediction)
        elif(self.kernel == "Gaussian"):
            num_test_samples = x.shape[0]
            classes = self.classes
            h = self.h
            
            predictions = []
            for i in range(num_test_samples):
                x_ = self.x_train - x[i, :]
                x_ = np.exp(-(np.linalg.norm(x_, axis = 1) ** 2) / (2 * h ** 2))
                
                count = 0
                prediction = classes[0]
                for j in classes:
                    temp = self.y_train == j
                    temp = x_[temp]
                    t_ = temp.sum()
                    if(t_ > count):
                        count = t_
                        prediction = j

                predictions.append(prediction)
            
        return np.array(predictions)

class LogisticRegression:
    classes = None
    weights = None
    regularize = None

    def __init__(self, regularize = False):
        self.regularize = regularize

    def Fit(self, x, y, learning_rate = 0.1, iterations = 10000, alpha = 0.1):
        classes = list(set(y))
        num_samples = x.shape[0]
        num_features = x.shape[1]
        x_ = x.T

        weights = {}
        
        for i in classes:
            y_ = (y == i).astype(int)

            if(not self.regularize):
                alpha = 0
            
            W = np.random.randn(1, num_features)
            b = np.random.randn(1)

            for j in range(iterations):
                h = Sigmoid(np.matmul(W, x_) + b)

                W_grad = ((h - y_) * x_).sum(axis = 1) / num_samples + 2 * alpha * W
                b_grad = (h - y_).sum(axis = 1) / num_samples

                W -= learning_rate * W_grad
                b -= learning_rate * b_grad

            weights[i] = {
                "W": W,
                "b": b
            }
            
            self.classes = classes
            self.weights = weights
    
    def Predict(self, x, threshold = 0.5):
        classes = self.classes
        weights = self.weights
        num_samples = x.shape[0]
        x_ = x.T

        predictions = {}
        concat_ = False
        for i in classes:
            W = weights[i]["W"]
            b = weights[i]["b"]

            temp_ = Sigmoid(np.matmul(W, x_) + b).reshape(num_samples)
            if(not concat_):
                q = temp_.reshape(num_samples, 1)
                concat_ = True
            else:
                q = np.concatenate([q, temp_.reshape(num_samples, 1)], axis = 1)
        
        # thresholds_array = np.linspace(0, 1, num = 10)
        # pl = np.zeros([1000, 4])
        # index = 0
        # for i in thresholds_array:
        #     for j in thresholds_array:
        #         for k in thresholds_array:
        #             q_ = np.array([i, j, k]) * q
                    
        #             q_ = np.argmax(q_, axis = 1)
        #             classes = np.array(classes)
        #             q_ = classes[q_]

        #             pl[index, :] = np.array([i, j, k, Accuracy(q_, y)])
        #             index += 1

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection = "3d")
        # ax.scatter(pl[:, 0], pl[:, 1], pl[:, 2], c = pl[:, 3], cmap = "hot")

        q = np.argmax(q, axis = 1)
        classes = np.array(classes)
        predictions = classes[q]
        
        return predictions

class SoftmaxRegression:
    classes = None
    weights = None

    def Fit(self, x, y, learning_rate = 0.1, iterations = 10000):
        classes = list(set(y))
        num_samples = x.shape[0]
        x = np.concatenate([np.ones([num_samples, 1]), x], axis = 1)
        num_features = x.shape[1]
        num_classes = len(classes)
        x_ = x.T

        y_ = np.zeros([num_classes, num_samples])
        for i in range(num_classes):
            temp = y == classes[i]
            y_[i, temp] = 1
        
        weights = {}
        W = np.random.randn(num_classes, num_features)
        
        for i in range(iterations):
            temp = np.exp(np.matmul(W, x_))
            z = temp.sum(axis = 0).reshape(1, num_samples, 1)

            grad1 = temp ** 2
            grad1 = grad1.T.reshape(1, num_samples, num_classes)
            grad1 = grad1 * x_.reshape(num_features, num_samples, 1) / z ** 2

            grad2 = y_ * temp
            grad2 = grad2.T.reshape(1, num_samples, num_classes)
            grad2 = grad2 * x_.reshape(num_features, num_samples, 1) / z

            W_grad = (grad1 - grad2).sum(axis = 1).T / num_samples
            W -= learning_rate * W_grad

        weights["W"] = W
        self.weights = weights
        self.classes = classes

        temp = Softmax(np.matmul(W, x_))
        temp = np.argmax(temp, axis = 0)
        classes = np.array(classes)
        predictions = classes[temp]

        return predictions

    def Predict(self, x):
        classes = self.classes
        weights = self.weights
        num_samples = x.shape[0]
        x = np.concatenate([np.ones([num_samples, 1]), x], axis = 1)
        x_ = x.T

        W = weights["W"]

        temp = Softmax(np.matmul(W, x_))
        temp = np.argmax(temp, axis = 0)
        classes = np.array(classes)
        predictions = classes[temp]

        return predictions

class LinearRegression:
    regularize = None
    weights = None
    training_error = None

    def __init__(self, regularize = False):
        self.regularize = regularize
    
    def Fit(self, x, y, alpha = 0.1):
        num_samples = x.shape[0]
        x = np.concatenate([np.ones([num_samples, 1]), x], axis = 1)
        num_features = x.shape[1]
        x_ = x.T
        y_ = y.reshape(1, num_samples)

        if(not self.regularize):
            alpha = 0
        
        weights = {}
        W = np.matmul(y_, np.matmul(x_.T, np.linalg.inv(np.matmul(x_, x_.T) + alpha * np.identity(num_features))))
        
        weights["W"] = W

        predictions = np.matmul(W, x_)
        self.training_error = ((predictions - y) ** 2).sum()
        self.weights = weights
        
        return predictions.reshape(num_samples)
    
    def Predict(self, x):
        weights = self.weights

        num_samples = x.shape[0]
        x = np.concatenate([np.ones([num_samples, 1]), x], axis = 1)
        x_ = x.T

        W = weights["W"]
        predictions = np.matmul(W, x_)

        return predictions