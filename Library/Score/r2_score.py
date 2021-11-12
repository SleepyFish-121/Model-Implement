import numpy as np


def r2_score(y_true: np.array, y_pred: np.array):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
