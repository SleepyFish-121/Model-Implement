import numpy as np


class KMeans():
    def __init__(self, n_clusters: int = 8, max_iter: int = 300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X: np.array):
        size_of_input = X.shape[0]
        random_choice = np.random.randint(size_of_input, size=self.n_clusters)
        self.cluster_centers = X[random_choice]
        self.labels = np.zeros(size_of_input)
        prior_cluster_centers = self.cluster_centers

        iter = 0
        while ((prior_cluster_centers - self.cluster_centers).any() or iter < self.max_iter):
            for i in range(size_of_input):
                self.labels[i] = np.argmin(np.sum((X[i] - self.cluster_centers) ** 2, axis=1))

            for i in range(self.n_clusters):
                self.cluster_centers[i] = np.mean(X[self.labels == i], axis=0)
            iter = iter + 1

        self.labels = self.labels.astype('int')

        return self

    def predict(self, X: np.array):
        labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            labels[i] = np.argmin(np.sum((X[i] - self.cluster_centers) ** 2, axis=1))
        return labels.astype('int')
