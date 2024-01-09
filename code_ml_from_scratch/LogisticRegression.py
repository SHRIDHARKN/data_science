import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression():

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weigths = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weigths = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weigths)+self.bias
            predictions = sigmoid(linear_pred)
            dw = (1/n_samples)*(np.dot(X.T, (predictions-y)))
            db = (1/n_samples)*(np.sum(predictions-y))
            self.weigths = self.weigths-self.lr*dw
            self.bias = self.bias-self.lr*db

    def predict(self, X):
        linear_pred = np.dot(X, self.weigths)+self.bias
        predictions = sigmoid(linear_pred)
        class_predictions = [0 if x <= 0.5 else 1 for x in predictions]
        return class_predictions
