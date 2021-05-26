import numpy as np

from ..base.BaseEstimator import BaseEstimator


class Ridge(BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super()

    def fit(self, X: np.array, y: np.array):
        self.X = X
        self.y = y
        self.X = np.hstack([np.array([1] * self.X.shape[0]).reshape(-1, 1),
                            self.X])  # Adding Constant
        self.b = np.linalg.inv(self.X.T @ self.X + self.alpha * np.eye(self.X.shape[1])) \
                 @ self.X.T @ self.y  # Using OLS
        return self

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.b
