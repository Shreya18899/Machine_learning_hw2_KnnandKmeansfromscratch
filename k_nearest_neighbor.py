import numpy as np 
import pandas as pd
import sklearn
import math

from distances import euclidean, cosim

from sklearn.decomposition import PCA

class KNearestNeighbor():    
    def __init__(self, n_neighbors=None, distance_measure='euclidean'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'cosim', use cosine similarity.


        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'cosim'. This is the distance measure
                that will be used to compare features to produce labels. 
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure

    def scale_greyscale_features(self, features):
        """Scale the greyscale values (0-255) to be between 0 and 1."""
        
        # for row in features:
        #     for i in range(len(row)):
        #         row[i] = row[i] / 255

        # return features

        return features / 255


    def pca_fit_transform(self, X, n_components=75):
        """Fit the PCA model to X and transform X with the PCA model."""
        self.pca = PCA(n_components=n_components, random_state=0)
        self.pca.fit(X)
        X = self.pca.transform(X)
        return X

    def pca_transform(self, X):
        """Transform X with the PCA model."""
        X = self.pca.transform(X)
        return X


    def fit(self, features, targets):
        """
        Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """

        self.features = features
        self.targets = targets
        self.set_default_label_ordering()

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        """
        This function does the same as fit, but it also finds the best k using the validation set.
        """
        
        self.features = X_train
        self.targets = y_train
        self.set_default_label_ordering()


        best_k = None
        best_acc = 0
        # if number of neighbors or K is not specified, find best k
        if self.n_neighbors is None:
            # k_to_check is 2, and up to 3 evenly spaced ints between 3 and the sqrt of the number of features
            # generates values to check for k from 2 to sqrt(2000) = 44 [checks 4 in total 2 + 3 evenly spaced numbers from
            # 3 to 28
            k_to_check = [2] + np.unique(np.linspace(3, math.sqrt(len(self.features)), 3, dtype=int)).tolist()
            print('k\'s to test: ', k_to_check)
            for k in k_to_check:
                preds = self.predict_internal(X_val, ignore_first=False, n_neighbors=k)
                acc = np.mean(preds == y_val)
                print('        trying k={} --> accuracy={}'.format(k, acc))
                if acc > best_acc:
                    best_k = k
                    best_acc = acc
            self.n_neighbors = best_k


        
    def set_default_label_ordering(self):
        """
        This function sets the default label ordering for breaking ties.
        The label are ordered by the frequency of their occurrence in the training set. This will be used later to break ties, where if there is a tie between two labels, the label that shows up first in the default ordering will be chosen as the prediction.
        If two labels show up equally in the training set, we arbitrarily prefer the label that is larger (e.g. B > A) for consistency
        Tie Scenario:

        Suppose the labels of the three nearest neighbors are: [A, B, C].
        The counts are: A: 1, B: 1, C: 1.
        Tie: No single label appears more frequently than others; all have the same count (1 each).
        [Note: tie scenario will not occur if k=3 and if the labels are binary]
        """

        # Count the number of occurrences of each label in the training set
        label_counts = {}
        for label in self.targets:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        # sort the labels according to 1. frequency of occurrence, 2. order of label
        self.default_label_ordering = sorted(label_counts, key=lambda label: (label_counts[label], label), reverse=True)



    def get_mode_with_order(self, lst):
        """Return the mode of a list. If there are multiple modes, return the first one in the order list."""
        counts = {}
        for item in lst:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
        max_count = max(counts.values())
        modes = [k for k, v in counts.items() if v == max_count]
       
        for item in self.default_label_ordering:
            if item in modes:
                return item

        return modes[0]
        

    def sort_return_indices(self, arr):
        """For a given array, returns the indices of the sorted array (smallest to largest)"""
        indices = list(range(len(arr)))
        indices.sort(key=lambda i: arr[i])
        return indices

    
    def predict_internal(self, samples, ignore_first=False, n_neighbors=None):
        """
        Samples - validation set/test set depending on what is passed in n_neighbors
        """
        # n_neighbors is None if we are doing real life prediction
        # n_neighbors is not None if we are doing validation on the test set
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        labels = []

        # for each test example
        for test_sample in samples:
            distances = []
            for neighbor in self.features:
                if self.distance_measure == 'euclidean':
                    distance = euclidean(test_sample, neighbor)
                elif self.distance_measure == 'cosim':
                    distance = 1 - cosim(test_sample, neighbor) # 1 - similarity so that we choose the k smallest values, consistent with euclidean distance
                distances.append(distance)


            # get smallest distances and their indices
            if ignore_first:
                closest_neighbors = self.sort_return_indices(distances)[1:n_neighbors+1]
            else:
                closest_neighbors = self.sort_return_indices(distances)[0:n_neighbors]


            pred = self.get_mode_with_order(self.targets[closest_neighbors])
            labels.append(pred)

        return labels

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """

        return self.predict_internal(features, ignore_first, n_neighbors=None)

    


    

    def classification_report(self, y_true, y_pred):
        return sklearn.metrics.classification_report(y_true, y_pred)
