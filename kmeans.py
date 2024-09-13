import sklearn
import numpy as np
from sklearn.decomposition import PCA
from distances import euclidean, cosim  # euclidean_kmeans


class KMeans():
    def __init__(self, n_clusters, distance_measure, max_iter):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.distance_measure = distance_measure
        self.max_iter = max_iter
        self.means = None

    # def scale_greyscale_features(self, features):
    #     """Scale the data to be between 0 and 1."""
    #     for row in features:
    #         for i in range(len(row)):
    #             row[i] = row[i] / 255
    #     return features

    def pca_fit_transform(self, X):
        """Fit the PCA model to X and transform X with the PCA model."""
        self.pca = PCA(n_components=0.95)
        self.pca.fit(X)
        X = self.pca.transform(X)
        return X

    def pca_transform(self, X):
        """Transform X with the PCA model."""
        X = self.pca.transform(X)
        return X

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        n_samples, n_features = features.shape
        # initialize means randomly
        # self.means will be a vector of shape n_clusters * 784 (reduced with pca)
        print("K Means clustering started")
        self.means = features[np.random.choice(n_samples, self.n_clusters, replace=False)]
        # print("Random means initialized")
        # print("Distance calculation with metric : ", self.distance_measure)
        for iteration in range(self.max_iter):
            # Calculate the euclidean distance between each training sample with all cluster centroids
            # returns distances of n_samples * dim (e.g. 784) and self.means = n_clusters * 784
            # returns 200 * 5 as distance array, where each observation has distance computed from n_clusters
            if self.distance_measure == 'euclidean':
                distances = [[euclidean(feat, mean) for mean in self.means] for feat in features]
            elif self.distance_measure == 'cosim':
                # 1 - cosim since cosim will give values from -1 to 1, making it a more intuitive distance metric for computation
                # if cosim(feat, mean) = 1, distance = 0, the point and means coincide
                # if cosim(feat, mean) = 1, distance = 2, maximum distance from the point
                # distances will be made positive, which inturn is more intuitive
                distances = [[1 - cosim(feat, mean) for mean in self.means] for feat in features]
            # distances2 = np.linalg.norm(features[:, np.newaxis] - self.means, axis=2)
            # take the index of the minimum value of a sample with the cluster centroid and assign it that label
            labels = np.argmin(distances, axis=1)

            # Count the number of labels assigned in this iteration
            # unique, counts = np.unique(labels, return_counts=True)
            # print(f"Iteration {iteration + 1} label counts: {dict(zip(unique, counts))}")

#
            # Handling empty clusters
            new_means = np.zeros_like(self.means)
            for k in range(self.n_clusters):
                if not np.any(labels == k):
                    # Reinitialize the mean of the empty cluster to a random data point
                    self.means[k] = features[np.random.choice(n_samples)]
                else:
                    # Update cluster centers by computing the mean of the samples
                    # if 5 points are assigned to a cluster, calculate the mean of the 5 points to form a centroid of
                    # dimension 1*784
                    new_means[k] = features[labels == k].mean(axis=0)
            # new_means = np.array([features[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # Check for convergence, if means do not change over 2 iterations k means has converged
            if np.all(new_means == self.means):
                print("CONVERGED")
                break
            self.means = new_means
        else:
            print("Max iterations reached without convergence")
        # raise NotImplementedError()

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        # use the updated cluster centers (means) to predict the new values
        if self.distance_measure == 'euclidean':
            distances = [[euclidean(feat, mean) for mean in self.means] for feat in features]
        elif self.distance_measure == 'cosim':
            distances = [[1 - cosim(feat, mean) for mean in self.means] for feat in features]
        labels = np.argmin(distances, axis=1)
        return labels
        # raise NotImplementedError()

    def classification_report(self, y_true, y_pred):
        return sklearn.metrics.classification_report(y_true, y_pred)

