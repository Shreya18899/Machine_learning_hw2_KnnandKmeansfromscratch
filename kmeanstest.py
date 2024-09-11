# File to test the kmeans algorithm

import pandas as pd
import numpy as np
import starter
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from k_nearest_neighbor import KNearestNeighbor
from kmeans import KMeans
from softkmeans import SoftKMeans
from soft_kmeans2 import SoftKMeans2
from sklearn.cluster import KMeans as SKMeans
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score


def read_initial_data(filename):
    """
    Read the initial data for k means algorithm
    """
    data = pd.read_csv(f"{filename}", header=None)
    # remove null column at end
    data = data.iloc[:, :-1]
    # data = data.sample(frac=0.1, random_state=1).reset_index(drop=True)
    # Taking the data ignoring the labels column
    X, y = data.iloc[:, 1:], data.iloc[:, 0]
    return X, y


def analyze_dataframe(_df):
    """
    analyze_dataframe takes a dataframe and returns a dataframe with the following
    """
    result = pd.DataFrame(
        columns=["Column", "Type", "#NA", "%NA", "#Unique"]
    )  # set up dataframe

    for col in _df.columns:  # for each column, compute the desired calculations
        row = [
            col,
            _df[col].dtype,
            _df[col].isnull().sum(),
            "{:.2%}".format(_df[col].isnull().sum() / len(_df)),
            _df[col].nunique(),
        ]
        result.loc[len(result)] = row  # add calculations to row of result dataframe

    return result


def assign_cluster_labels(clusters, actual_labels):
    """
    Assigns labels to clusters based on the most common actual label in each cluster.

    Parameters:
    - clusters (list or np.ndarray): Cluster assignments for each data point.
    - actual_labels (list or np.ndarray): True labels for each data point.

    Returns:
    - dict: Mapping from cluster index to actual label.
    """
    assigned_labels = {}
    for cluster_id in np.unique(clusters):
        # Find actual labels for samples within this cluster
        labels_in_cluster = actual_labels[clusters == cluster_id]
        # Assign the most common actual label to this cluster
        assigned_label = np.bincount(labels_in_cluster).argmax()
        assigned_labels[cluster_id] = assigned_label
    return assigned_labels


def test_kmeans(train_file, test_file, valid_file):
    train_x, actual_labels_train = read_initial_data(train_file)
    test_x, actual_labels_test = read_initial_data(test_file)
    valid_x, actual_labels_valid = read_initial_data(valid_file)

    # print("Actual labels")
    # print(actual_labels_train, actual_labels_test)

    # turn data into list of lists
    # X = X.values.tolist()
    train_x = np.array(train_x)
    test_x = np.array(test_x)

    # Create an object of the class
    # Implement a max iterations for convergence
    # Set k=10 for mnist data
    model = KMeans(n_clusters=10, distance_measure="cosim", max_iter=300)

    # Scale values assuming grayscale
    ss = MinMaxScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.transform(test_x)
    # train_x = model.scale_greyscale_features(train_x)
    # test_x = model.scale_greyscale_features(test_x)

    print("Dimension before pca", train_x.shape)
    # Dimensionality reduction
    train_x = model.pca_fit_transform(train_x)
    test_x = model.pca_transform(test_x)
    print("Dimensionality reduction performed ", train_x.shape)

    # Fit the model
    model.fit(train_x)
    print("Model fitted with specified clusters")

    # Generate predictions with cluster centroids
    clusters_test = model.predict(test_x)
    print(clusters_test)
    print(actual_labels_train)
    print("Prediction labels generated")

    # After your KMeans fit method:
    clusters_train = model.predict(train_x)
    print("Printing clusters_train")
    print(set(clusters_train))
    print(actual_labels_train.unique())
    label_map = assign_cluster_labels(clusters_train, actual_labels_train)

    estimated_labels_train = np.array([label_map[cluster] for cluster in clusters_train])

    # clusters_test = model.predict(test_x)
    estimated_labels_test = np.array([label_map[cluster] for cluster in clusters_test])

    accuracy_train = np.mean(estimated_labels_train == actual_labels_train)
    accuracy_test = np.mean(estimated_labels_test == actual_labels_test)
    print(accuracy_train, accuracy_test)
    print(model.classification_report(actual_labels_test, estimated_labels_test))


def test_softkmeans(train_file, test_file, valid_file):
    train_x, actual_labels_train = read_initial_data(train_file)
    test_x, actual_labels_test = read_initial_data(test_file)
    valid_x, actual_labels_valid = read_initial_data(valid_file)

    train_x = np.array(train_x)
    test_x = np.array(test_x)

    # Create an object of the class
    # Implement a max iterations for convergence
    # Set k=10 for mnist data
    beta = [0.01]
    softkmeans = SoftKMeans(n_clusters=10, distance_measure="cosim", max_iter=300, beta=0.01)
    # Scale values assuming grayscale
    mm_sc = MinMaxScaler()
    train_x = mm_sc.fit_transform(train_x)
    test_x = mm_sc.transform(test_x)
    # train_x = softkmeans.scale_greyscale_features(train_x)
    # test_x = softkmeans.scale_greyscale_features(test_x)

    print("Dimension before pca", train_x.shape)
    # Dimensionality reduction
    train_x = softkmeans.pca_fit_transform(train_x)
    test_x = softkmeans.pca_transform(test_x)
    print("Dimensionality reduction performed ", train_x.shape)

    # Fit the model
    softkmeans.fit(train_x)
    print("Model fitted with specified clusters")

    # Generate predictions with cluster centroids
    clusters_test = softkmeans.predict(test_x)
    print("Prediction labels generated")

    clusters_train = softkmeans.predict(train_x)
    label_map = assign_cluster_labels(clusters_train, actual_labels_train)
    estimated_labels_train = np.array([label_map[cluster] for cluster in clusters_train])
    estimated_labels_test = np.array([label_map[cluster] for cluster in clusters_test])

    accuracy_train = np.mean(estimated_labels_train == actual_labels_train)
    accuracy_test = np.mean(estimated_labels_test == actual_labels_test)
    print(accuracy_train, accuracy_test)


#
# def sklearnKmeans(train):
#     sklearnmodel = SKMeans(n_clusters=5).fit(train)
#     p = sklearnmodel.labels_
#     print(set(p))


if __name__ == "__main__":
    # pass the training and testing csvs
    test_kmeans(train_file="train.csv", test_file="test.csv", valid_file="valid.csv")
    print("Start testing soft k means")
    # test_softkmeans(train_file="train.csv", test_file="test.csv", valid_file="valid.csv")
    # test_softkmeans2(train_file="train.csv", test_file="test.csv", valid_file="valid.csv")

    # sklearnKmeans(train)
