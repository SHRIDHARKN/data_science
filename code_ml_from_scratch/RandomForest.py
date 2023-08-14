import numpy as np
from collections import Counter
from DecisionTree import DecisionTreeClassifier

class RandomForestClassifier:

    def __init__(self,n_estimators=10,max_depth=8,min_samples_split=2,
                n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
    
    def _bootstrap_samples(self,X,y):
        n_samples = X.shape[0]
        idxs  = np.random.choice(n_samples,n_samples,replace=True)
        return X[idxs],y[idxs]

    def _majority_vote(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def fit(self,X,y):
        self.trees=[]
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split,
                                            n_features=self.n_features)

            X_sample,y_sample = self._bootstrap_samples(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)                                

    def predict(self,X):

        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = np.swapaxes(predictions,0,1)
        return np.array([self._majority_vote(pred) for  pred in predictions])

    def predict_proba_(self,X):
        predictions = np.array([tree.predict_proba_(X) for tree in self.trees])
        predictions = np.swapaxes(predictions,0,1)
        return np.mean(predictions,axis=1)
    