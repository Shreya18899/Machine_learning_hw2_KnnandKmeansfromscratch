import time
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
# our code
from distances import euclidean, cosim
from k_nearest_neighbor import KNearestNeighbor
from kmeans import KMeans
from softkmeans import SoftKMeans
# for PCA EDA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# for visualizing confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_data(file_name, frac=1):
    '''
    Loads in a data file, returns X, y. Can take a sample with frac.
    '''
    data = pd.read_csv(file_name, header=None)
    data = data.iloc[:, :-1]  # remove null column at end
    if frac != 1:
        data = data.sample(frac=frac, random_state=1).reset_index(drop=True)
    X, y = data.iloc[:, 1:], data.iloc[:, 0]
    return X, y


def plot_confusion_matrix(y_true, y_pred, algorithm, data, distance_metric):
    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    # Use seaborn to create a heatmap
    sns.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title(f'Confusion Matrix for {algorithm} on {data} with {distance_metric}')

    # Plot the heatmap
    sns.heatmap(cm, annot=True, fmt='.2f')
    # , xticklabels=class_names, yticklabels=class_names)

    # Add xlabel and ylabel
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save the figure
    plt.savefig(f'confusion_matrix_{algorithm}_{data}_{distance_metric}.png')

    # Show the plot
    # plt.show()


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


def test_euclidean():
    """
    Test euclidean distance
    Calculating Distances:
        numpy_euclidean calculates the Euclidean distance using NumPy.
        our_euclidean calls your custom euclidean function.
    Assertion: It checks if the difference between the two calculated distances is within a very small tolerance (0.0001)
    """
    X, _ = load_data('valid.csv')

    print("Printing valid ", X.shape)

    v1 = X.iloc[0, :].values
    v2 = X.iloc[1, :].values

    numpy_euclidean = np.sqrt(np.sum((v1 - v2) ** 2))
    our_euclidean = euclidean(v1, v2)

    assert abs(our_euclidean - numpy_euclidean) < 0.0001
    print('euclidean test case passed!')


def test_cosim():
    """
    Test cosine similarity
    Calculating Distances:
        numpy_cosine calculates the Cosine similarity using NumPy.
        our_euclidean calls your custom cosine function.
    Assertion: It checks if the difference between the two calculated distances is within a very small tolerance (0.0001)
    """
    X, _ = load_data('valid.csv')

    v1 = X.iloc[0, :].values
    v2 = X.iloc[1, :].values

    our_cosim = cosim(v1, v2)
    numpy_cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert abs(our_cosim - numpy_cosine) < 0.0001
    print('cosim test case passed!')


def pca_eda():
    """
    Exploratory data analysis - graphing cumulative explained variance from PCA on training set
    """
    X, _ = load_data(file_name='train.csv')

    pca = PCA()
    X_pca = pca.fit(X)

    # Get the explained variance ratio for all principal components
    explained_var_ratio = pca.explained_variance_ratio_

    # Create a DataFrame to display the results
    explained_var_df = pd.DataFrame({'PCA Component': range(1, len(explained_var_ratio) + 1),
                                     'Explained Variance Ratio': explained_var_ratio})
    cumulative_explained_var = explained_var_ratio.cumsum()

    # Graph PCA explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_explained_var) + 1), cumulative_explained_var, marker='o', linestyle='-')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid()
    # plt.show()
    plt.savefig('pca_eda.png')

    # Zoom into 30-100
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_explained_var) + 1), cumulative_explained_var, marker='o', linestyle='-')
    plt.xlabel('Number of Principal Components (30-100)')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance (Principal Components 30-100)')
    plt.grid()
    plt.xlim(30, 100)
    # plt.show()
    plt.savefig('pca_eda_zoomed.png')


def run_KNN(title, distance_measure='euclidean', n_neighbors=None):
    """
    Train and test KNN with specified distance measure and n_neighbors.
    If n_neighbors is None, then the model will automatically find the best k using the validation set.
    Arguments:
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. 
            n_neighbors {int} -- Number of neighbors to use for prediction. If None, then the model will automatically find the best k using the validation set.
    """
    X_train, y_train = load_data('train.csv')
    X_val, y_val = load_data('valid.csv')
    X_test, y_test = load_data('test.csv')

    print_str = '========{}========'.format(title)
    print('=' * len(print_str))
    print(print_str)
    print('=' * len(print_str))

    knn = KNearestNeighbor(n_neighbors=n_neighbors, distance_measure=distance_measure)

    # Assuming grayscale
    X_train = knn.scale_greyscale_features(X_train)
    X_val = knn.scale_greyscale_features(X_val)
    X_test = knn.scale_greyscale_features(X_test)

    # Dimensionality reduction
    X_train = knn.pca_fit_transform(X_train, n_components=75).tolist()
    X_val = knn.pca_transform(X_val).tolist()
    X_test = knn.pca_transform(X_test).tolist()

    # Automatically find best k using validation set
    if n_neighbors is None:
        knn.fit_with_validation(X_train, y_train, X_val, y_val)
    else:
        knn.fit(X_train, y_train)
    print('\n  =>Using k={}'.format(knn.n_neighbors))
    y_pred = knn.predict(X_test)

    print('\n             ==========Classification Report=========='.format(knn.n_neighbors))
    print(knn.classification_report(y_test, y_pred))
    print('\n\n')
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title('Confusion Matrix for {}'.format(title))
    plt.savefig('confusion_matrix_{}.png'.format(title))


def run_Kmeans(distance_measure, max_iter):
    """
    Train and test Kmeans with specified distance measure and max_iterations.
    Arguments:
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'cosine'.
            max_iter - used for convergence to find solution
    """
    X_train, y_train = load_data('train.csv')
    X_val, y_val = load_data('valid.csv')
    X_test, y_test = load_data('test.csv')

    print_str = '========running KMeans with {} distance measure========'.format(distance_measure)
    print('=' * len(print_str))
    print(print_str)
    print('=' * len(print_str))

    if distance_measure == "cosim":
        print("Cosine similarity may take longer to converge")

    kmeans = KMeans(n_clusters=10, distance_measure=distance_measure, max_iter=max_iter)
    # knn = KNearestNeighbor(n_neighbors=n_neighbors, distance_measure=distance_measure)

    # Use MinMax Scaling
    mm_scaler = MinMaxScaler()
    X_train = mm_scaler.fit_transform(X_train)
    X_val = mm_scaler.transform(X_val)
    X_test = mm_scaler.transform(X_test)

    prev_shape = X_train.shape[1]
    # Dimensionality reduction
    X_train = kmeans.pca_fit_transform(X_train)
    X_val = kmeans.pca_transform(X_val)
    X_test = kmeans.pca_transform(X_test)

    print("Data downsized from ", prev_shape, "to ", X_train.shape[1])
    # Fit the model
    kmeans.fit(X_train)

    # Generate predictions with cluster centroids
    clusters_test = kmeans.predict(X_test)
    clusters_train = kmeans.predict(X_train)
    clusters_val = kmeans.predict(X_val)
    label_map = assign_cluster_labels(clusters_train, y_train)

    estimated_labels_train = np.array([label_map[cluster] for cluster in clusters_train])
    estimated_labels_test = np.array([label_map[cluster] for cluster in clusters_test])
    estimated_labels_val = np.array([label_map[cluster] for cluster in clusters_val])

    accuracy_train = np.mean(estimated_labels_train == y_train)
    accuracy_test = np.mean(estimated_labels_test == y_test)
    accuracy_val = np.mean(estimated_labels_val == y_val)

    print('\n==========Classication Report==========')
    print("Using mode to assess K means effectiveness ")
    print("Testing accuracy of K means : ", accuracy_test, "\n")
    print("Validation accuracy of K means : ", accuracy_val, "\n")
    print(kmeans.classification_report(y_test, estimated_labels_test))
    print(kmeans.classification_report(y_val, estimated_labels_val))

    print("Confusion matrix getting generated")
    # print(kmeans.confusion_matrix(y_test, estimated_labels_test))
    plot_confusion_matrix(y_test, estimated_labels_test, "KMeans", "Test", distance_measure)
    plot_confusion_matrix(y_val, estimated_labels_val, "KMeans", "Valid", distance_measure)


def run_SoftKmeans(distance_measure, max_iter, beta_values):
    """
    Train and test Kmeans with specified distance measure and max_iterations.
    Arguments:
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'cosine'.
            max_iter - used for convergence to find solution
    """
    X_train, y_train = load_data('train.csv')
    X_val, y_val = load_data('valid.csv')
    X_test, y_test = load_data('test.csv')

    print_str = '\n========running Soft KMeans with {} distance measure========'.format(distance_measure)
    print('=' * len(print_str))
    print(print_str)
    print('=' * len(print_str))

    if distance_measure == "cosim":
        print("Cosine similarity may take longer to converge")

    # Use MinMax Scaling
    mm_scaler = MinMaxScaler()
    X_train = mm_scaler.fit_transform(X_train)
    X_val = mm_scaler.transform(X_val)
    X_test = mm_scaler.transform(X_test)
    # softkmeanstmp = SoftKMeans(n_clusters=10, distance_measure=distance_measure, max_iter=max_iter, beta=0.1)

    # X_train = softkmeanstmp.pca_fit_transform(X_train)
    # X_val = softkmeanstmp.pca_transform(X_val)
    # X_test = softkmeanstmp.pca_transform(X_test)
    # print("Dimensionality reduction performed ", X_train.shape)

    print("Hyperparameters which softkmeans will be tested with = ", beta_values)
    for beta in beta_values:
        softkmeans = SoftKMeans(n_clusters=10, distance_measure=distance_measure, max_iter=max_iter, beta=beta)
        print("\nFitting K means with hyperparamater beta = ", beta)
        # Fit the model
        softkmeans.fit(X_train)

        # Generate predictions with cluster centroids
        clusters_test = softkmeans.predict(X_test)
        clusters_train = softkmeans.predict(X_train)
        clusters_val = softkmeans.predict(X_val)
        label_map = assign_cluster_labels(clusters_train, y_train)

        estimated_labels_train = np.array([label_map[cluster] for cluster in clusters_train])
        estimated_labels_test = np.array([label_map[cluster] for cluster in clusters_test])
        estimated_labels_val = np.array([label_map[cluster] for cluster in clusters_val])

        accuracy_train = np.mean(estimated_labels_train == y_train)
        accuracy_test = np.mean(estimated_labels_test == y_test)
        accuracy_val = np.mean(estimated_labels_val == y_val)

        print('\n==========Classication Report==========')
        print("Using mode to assess K means effectiveness ")
        print(f"Testing accuracy of K means with {beta} : ", accuracy_test)
        print(softkmeans.classification_report(y_test, estimated_labels_test))
        print("\n")


if __name__ == "__main__":
    test_euclidean()
    print()
    test_cosim()
    print()
    # pca_eda()

    start_time = time.time()

    # run KNN with euclidean
    print("Running KNN with euclidean and K not specified")
    run_KNN(distance_measure='euclidean', n_neighbors=None, title='KNN Auto-k with euclidean')

    # # run KNN with cosim
    # run_KNN(distance_measure='cosim', n_neighbors=None, title='KNN Auto-k with cosim')
    #
    # # run KNN with specified k
    # run_KNN(distance_measure='euclidean', n_neighbors=2, title='KNN k=2 with euclidean')
    #
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time for knn's: {elapsed_time} seconds")
    #
    # st_kmeans = time.time()
    # print("K means starting with k = 10 clusters assuming MNIST dataset")
    # # run Kmeans with euclidean
    # run_Kmeans(distance_measure='euclidean', max_iter=100)
    # # run Kmeans with cosim
    # run_Kmeans(distance_measure='cosim', max_iter=100)
    # print(f"Elapsed time for k means : {time.time() - st_kmeans} seconds \n\n")
    #
    # st_skmeans = time.time()
    # print("Soft K means starting with k = 10 clusters assuming MNIST dataset")
    # # run soft Kmeans with cosim
    # run_SoftKmeans(distance_measure='cosim', max_iter=100, beta_values=[0.01, 0.02, 0.03, 0.05])
    # # run soft Kmeans with euclidean
    # run_SoftKmeans(distance_measure='euclidean', max_iter=100, beta_values=[0.01])
    # print(f"Elapsed time for soft k means : {time.time() - st_skmeans} seconds")
