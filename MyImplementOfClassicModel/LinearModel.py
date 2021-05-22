import numpy as np


class LinearRegreesion():
    def __init__(self, ):
        pass

    def fit(self, X: np.array, y: np.array, return_result: bool = True):
        self.X = X
        self.y = y
        if self.X.shape[0] != self.y.shape[0]:
            raise Exception("Unmatching X and y")
        self.X = np.hstack([np.array([1] * self.X.shape[0]).reshape(-1, 1), self.X])  # Adding Constant
        self.b = np.matmul(np.matmul(np.power(np.matmul(self.X.T, self.X), -1), self.X.T), self.y)  # Using OLS
        if return_result == True:
            return self, self.b
        else:
            return self

    def predict(self, X):
        X = np.hstack([np.array([1]).reshape(1, 1), X])
        return np.matmul(X, self.b).T
