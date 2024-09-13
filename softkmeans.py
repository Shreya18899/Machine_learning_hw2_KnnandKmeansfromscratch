import numpy as np
import sklearn
from sklearn.decomposition import PCA
from distances import euclidean, cosim


class SoftKMeans():
    def __init__(self, n_clusters, distance_measure, max_iter, beta):
        self.n_clusters = n_clusters
        self.distance_measure = distance_measure
        self.max_iter = max_iter
        self.beta = beta
        # self.learning_rate = learning_rate
        self.means = None

    def scale_greyscale_features(self, features):
        """Scale the greyscale values (0-255) to be between 0 and 1."""
        return features / 255

    def pca_fit_transform(self, X):
        """Fit the PCA model to X and transform X with the PCA model."""
        # self.pca = PCA(n_components=0.95)
        self.pca = PCA()
        self.pca.fit(X)
        X = self.pca.transform(X)
        return X

    def pca_transform(self, X):
        """Transform X with the PCA model."""
        X = self.pca.transform(X)
        return X

    def compute_distance(self, x, y):
        if self.distance_measure == 'euclidean':
            return euclidean(x, y)
        elif self.distance_measure == 'cosim':
            return cosim(x, y)
        else:
            raise ValueError("Invalid distance_measure")

    def fit(self, features):
        n_samples, n_features = features.shape
        # randomly initialize the cluster centroids
        self.means = features[np.random.choice(n_samples, self.n_clusters, replace=False)]
        # print("Random means initialized")

        for iteration in range(self.max_iter):
            # Compute fuzzy assignments
            distances = np.array([[self.compute_distance(feat, mean) for mean in self.means] for feat in features])
            # weights = np.exp(-self.beta * distances)
            # weights = np.exp(-self.beta * (distances**2))
            weights = 1 / (1 + (self.beta * distances)**3)

            # Divide by all the labels
            weights /= weights.sum(axis=1)[:, np.newaxis]
            # Update means
            new_means = np.dot(weights.T, features) / weights.sum(axis=0)[:, np.newaxis]

            tolerance = 1e-4  # Adjust this value as needed
            if np.allclose(new_means, self.means, atol=tolerance):
                print("CONVERGED")
                break

            self.means = new_means
        else:
            print("Max iterations reached without convergence")

    def predict(self, features):
        distances = np.array([[self.compute_distance(feat, mean) for mean in self.means] for feat in features])
        labels = np.argmin(distances, axis=1)
        # Count the number of labels assigned in this iteration
        unique, counts = np.unique(labels, return_counts=True)
        # print(f"label counts: {dict(zip(unique, counts))}")
        return labels

    def classification_report(self, y_true, y_pred):
        return sklearn.metrics.classification_report(y_true, y_pred)

