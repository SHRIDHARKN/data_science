import numpy as np
from collections import Counter


def eucledian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
    
class KNeighborsClassifier:
    
    def __init__(self,k=3):
        self.k = k
        
    def fit(self,X,y):
        self.X = X
        self.y = y
        
    def predict(self,X):
        predictions = [self._predictions(x) for x in X]
        return predictions
        
    def _predictions(self,x):
        
        # compute the distances for x
        distances = [eucledian_distance(x,x_train) for x_train in self.X]
        
        # get closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in k_indices]
        
        # majority vote
        most_common = Counter(k_nearest_labels).most_common()[0][0]
        return most_common
    

class KNeighborsRegressor:
    
    def __init__(self,k=3):
        self.k = k
        
    def fit(self,X,y):
        self.X = X
        self.y = y
        
    def predict(self,X):
        predictions = [self._predictions(x) for x in X]
        return predictions
        
    def _predictions(self,x):
        
        # compute the distances for x
        distances = [eucledian_distance(x,x_train) for x_train in self.X]
        
        # get closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in k_indices]
        
        return np.nanmean(k_nearest_labels)    # average of neighbors