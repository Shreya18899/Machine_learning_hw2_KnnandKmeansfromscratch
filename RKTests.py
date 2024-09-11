"""
doc string
"""

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from k_nearest_neighbor import KNearestNeighbor
from sklearn.metrics import classification_report

from distances import euclidean, cosim

def test_get_default_label_ordering():
    label_counts = {1: 1, 3: 100, 2: 100, 5: 2}
    
    default_label_ordering1 = sorted(label_counts, key=lambda label: (label_counts[label], label), reverse=True)
    

    label_counts = {1: 1, 2: 100, 3: 100, 5: 2}

    default_label_ordering2 = sorted(label_counts, key=lambda label: (label_counts[label], label), reverse=True)
    
    label_counts = {9: 100, 2: 100, 3: 100, 5: 2}

    default_label_ordering3 = sorted(label_counts, key=lambda label: (label_counts[label], label), reverse=True)
    
    label_counts = {3: 100, 9: 100, 2: 100, 5: 2}

    default_label_ordering4 = sorted(label_counts, key=lambda label: (label_counts[label], label), reverse=True)
    
    label_counts = {2: 100, 3: 100, 9: 100, 5: 2}

    default_label_ordering5 = sorted(label_counts, key=lambda label: (label_counts[label], label), reverse=True)

def read_train_data(frac=0.1):
    data = pd.read_csv("train.csv", header=None)
    data = data.iloc[:, :-1] # remove null column at end
    data = data.sample(frac=frac, random_state=1).reset_index(drop=True)
    X, y = data.iloc[:, 1:], data.iloc[:, 0]
    print('shape', X.shape)
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

def test_KNN1(distance_measure):

    _X, y = read_train_data(frac=0.1)
   

  
    model = KNearestNeighbor(n_neighbors=None, distance_measure=distance_measure)
    print('distance_measure', model.distance_measure)
      # turn data into list of lists
    X = _X.values.tolist()
    X = model.scale_greyscale_features(X)


    model.fit(X, y)
    print('n_neighbors', model.n_neighbors)

    preds = model.predict(X)
    y_true = y.copy()
    df = pd.DataFrame({'y_pred': preds, 'y_true': y_true})
    df.to_csv('train_preds.csv', index=False)
    # todo test softmax
    
    # compute accuracy
    accuracy = np.mean(preds == y)
    print('Accuracy', accuracy)

    if distance_measure == 'cosim':
        sklearn_distance_measure = 'cosine'
    else:
        sklearn_distance_measure = 'euclidean'

    sklearnKNN = KNeighborsClassifier(metric=sklearn_distance_measure, algorithm='brute')
    print('sklearnKNN', sklearnKNN.n_neighbors)
    _X = _X / 255
    #print(_X.iloc[0, :])
    sklearnKNN.fit(_X, y)
    sklearnPreds = sklearnKNN.predict(_X)
    sklearnAccuracy = np.mean(sklearnPreds == y)
    print('sklearn Accuracy', sklearnAccuracy)
    print(classification_report(y, preds))

def test_cosim():

    _X, y = read_train_data(frac=0.20)

    v1 = _X.iloc[0, :]
    v2 = _X.iloc[1, :]

    our_cosim = cosim(v1, v2)
    numpy_cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # print('our_cosim', our_cosim)
    # print('numpy_cosine', numpy_cosine)
    assert abs(our_cosim-numpy_cosine) < 0.0001


def test_sort_return_indices():

    arr = [5,4,3,2,1]
    knn = KNearestNeighbor(n_neighbors=3, distance_measure='euclidean')
    idx = knn.sort_return_indices(arr)
    assert arr == [1,2,3,4,5]


if __name__ == "__main__":

    #test_sort_return_indices()

    #test_cosim()

    test_KNN1(distance_measure='euclidean')

