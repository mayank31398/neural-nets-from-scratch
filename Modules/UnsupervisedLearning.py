import numpy as np
from Modules.Distances import Distance

class kmeansClassifier():
    cluster_centers = None
    distance_metric = None
    
    def __init__(self, distance_metric = "Euclidean"):
        self.distance_metric = distance_metric
    
    def Fit(self, x, k, iterations = 10000):
        num_samples = x.shape[0]
        num_features = x.shape[1]
        
        cluster_centers = np.random.randint(0, num_samples, size = [k])
        cluster_centers = x[cluster_centers].reshape(k, num_features)
        
        x_ = x.T
        x_ = x_.reshape(1, num_features, num_samples)
        
        for i in range(iterations):
            cluster_centers_ = cluster_centers.reshape(k, num_features, 1)
            dist = Distance(x_, cluster_centers_, metric = self.distance_metric, axis = 1)
            cluster = np.argmin(dist, axis = 0)
            
            for j in range(k):
                temp = cluster == j
                x_cluster_j = x[temp]
                cluster_centers[j] = x_cluster_j.sum(axis = 0) / (x_cluster_j.shape[0] + 1e-6)
            
        self.cluster_centers = cluster_centers
    
    def Predict(self, x):
        num_samples = x.shape[0]
        num_features = x.shape[1]
        
        cluster_centers = self.cluster_centers
        
        k = cluster_centers.shape[0]
        
        cluster_centers = cluster_centers.reshape(k, num_features, 1)
        x_ = x.T
        x_ = x_.reshape(1, num_features, num_samples)
        
        dist = Distance(x_, cluster_centers, metric = self.distance_metric, axis = 1)
        
        predictions = np.argmin(dist, axis = 0)
        
        return predictions
    
    def ComputeLoss(self, x):
        num_features = x.shape[1]
        
        cluster_centers = self.cluster_centers
        
        k = cluster_centers.shape[0]
        
        loss = 0
        predictions = self.Predict(x)
        for i in range(k):
            temp = predictions == i
            num_temp = temp.sum()
            x_ = x[temp, :].reshape(num_temp, num_features)
            cluster_centers_ = cluster_centers[i].reshape(1, num_features)
            loss += (Distance(x_, cluster_centers_, metric = self.distance_metric, axis = 1) ** 2).sum(axis = 0)
        
        return loss