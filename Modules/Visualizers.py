import numpy as np
import matplotlib.pyplot as plt

def VisualizePCA(x_train, P = [784, 700, 600, 500, 400, 300, 200, 100, 75, 50, 20, 10], index = 0):
    d = x_train.shape[1]
    
    means = x_train.mean(axis = 0, keepdims = True)
    x_ = x_train - means
    S = np.matmul(x_.T, x_)
    
    values, vectors = np.linalg.eigh(S)
    
    temp = values.argsort()
    temp = list(temp)
    temp.reverse()
    
    values = values[temp]
    vectors = vectors[:, temp]
    
    if(len(P) % 3 != 0):
        q = (int)(len(P) / 3) + 1
    else:
        q = (int)(len(P) / 3)
        
    x_train_ = x_train[index]
    fig, ax = plt.subplots(q, 5, figsize = [25, 25])
    for i in range(1, len(P) + 1):
        p = P[i - 1]
        
        vals = values[:p]
        vecs = vectors[:, :p]

        x_train_ = np.matmul(x_train[index, :], vecs)
    
        plt.subplot(q, 3, i)
        plt.title("p = " + str(p))
#         plt.imshow(x_train_.reshape(28, 28), cmap = "Greys_r")
        Z = np.matmul(x_train_, np.matmul(np.linalg.inv(np.matmul(vecs.T, vecs)), vecs.T))
        plt.imshow(Z.reshape(28, 28), cmap = "Greys_r")

def PCA(x_train, x_test, threshold = 0.95, p = 0):
    d = x_train.shape[1]
    
    means = x_train.mean(axis = 0, keepdims = True)
    x_ = x_train - means
    S = np.matmul(x_.T, x_)
    
    values, vectors = np.linalg.eigh(S)
    
    temp = values.argsort()
    temp = list(temp)
    temp.reverse()
    
    values = values[temp]
    vectors = vectors[:, temp]
    
    if(p == 0):
        var = 0
        total_var = values.sum()
        for i in range(d):
            var += values[i]
            if(var / total_var >= 0.95):
                p = i + 1
                break
    
    vals = values[:p]
    vecs = vectors[:, :p]
    
    x_train_ = np.matmul(x_train, vecs)
    x_test_ = np.matmul(x_test, vecs)
    
    ###########################################################################################
#     This is purely to see the images
    
#     plot = 0
    
#     plt.subplot(1, 2, 1)
#     plt.title("p = " + str(784))
#     plt.imshow(x_train[plot].reshape(28, 28), cmap = "Greys_r")
#     Z = np.matmul(x_train_[plot, :], np.matmul(np.linalg.inv(np.matmul(vecs.T, vecs)), vecs.T))
    
#     plt.subplot(1, 2, 2)
#     plt.title("p = " + str(p))
#     plt.imshow(Z.reshape(28, 28), cmap = "Greys_r")
    ###########################################################################################
    
    return x_train_, x_test_, p