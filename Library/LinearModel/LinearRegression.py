import numpy as np

from ..base.BaseEstimator import BaseEstimator


class LinearRegreesion(BaseEstimator):
    def __init__(self, ):
        super()

    def fit(self, X: np.array, y: np.array):
        self.X = X
        self.y = y
        self.X = np.hstack([np.array([1] * self.X.shape[0]).reshape(-1, 1),
                            self.X])  # Adding Constant
        self.b = np.linalg.inv(self.X.T @ self.X) \
                 @ self.X.T @ self.y  # Using OLS
        return self

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.b
