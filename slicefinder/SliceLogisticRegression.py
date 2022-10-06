from sklearn.linear_model import LogisticRegression

import numpy as np

class MyFakeLR(LogisticRegression):
    def __init__(self):
        
        self.classes_ = np.array([-1, 1])
        self.C = 1e6

    def fit(self, X, y):
        self.mean = y.mean()
    def predict_proba(self, X):
        results = np.ones((len(X), 2))
        results[:, 0] = results[:, 0] * (1-self.mean)
        results[:, 1] = results[:, 1] * (self.mean)
        return results
    def getModel(self):
        return self