from collections import Counter

import numpy as np

from ..base.BaseEstimator import BaseEstimator


def mode(input_list):
    return Counter(input_list).most_common(1)[0][0]


class KNeighborsClassifier(BaseEstimator):
    def __init__(self, n_neighbors=3, ):
        self.n_neighbors = n_neighbors
        super()

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        labels = np.zeros((X.shape[0], 1)).astype(self.y.dtype)
        for i in range(X.shape[0]):
            selected = np.argsort(np.sqrt(np.sum((X[i] - self.X) ** 2,
                                                 axis=1)), axis=0)[:self.n_neighbors]
            labels[i] = self.y[mode(selected)]
        return labels
