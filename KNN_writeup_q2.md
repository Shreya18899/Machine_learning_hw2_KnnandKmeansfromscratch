# Part 2 - KNN Writeup

## Implementation Details

1. Under the assumption this code will be used for the MNIST data set (features are greyscale images), we first divided all the feature values by 255, so that they now range between 0 and 1. This helps the computations run quicker. Since we know the min and max values for these features (0 and 255), we divided all values by 255 instead of using common scaling methods. Thus, all values in the test set will be guaranteed to be between 0 and 1. This is implemented in `scale_greyscale_features()`
2. We used PCA to reduce dimensionality as well. We determined from EDA on the training set that 75 was a good number of principal components to use, as it explained 88% of the variability in the training set.
    - See Appendix EDA - PCA on Training Set
3. To address ties, we created a default ordering of the labels in `set_default_label_ordering`. This gives preference to the labels that were most common in the overall data set. Second, if two labels show up with equal frequency, we arbitrarily choose the label that is larger (e.g. B > A) for consistency. This default label ordering is later used in the `get_mode_with_order()` function that returns the prediction for a sample based off its k-nearest neighbors. The`get_mode_with_order()` first checks for the mode, and if there is a tie, it chooses the mode that shows up first in the default ordering.
4. We gave the option to set the n_neighbors hyperparameter, but it can be set automatically as well (`KNearestNeighbor(n_neighbors=None)`). If it is unspecified, then the model will try a range of k's, and pick the one with the best accuracy (on the validation set). The range of k's is 2 and then up to three more evenly spaced integers between 3 and the square root of the number of samples in the training set. Our research suggested that the square root of the number of samples is an okay heuristic for choosing k, so we look for k's going up to that value. We limit to 3 more k's so that the algorithm doesn't take too long to fit.
    - Note: if auto-tuning is not used, the validation set is ignored in the main.py `run_KNN` function.
5. Note: When calculating cosine similarity, we stored 1 - the cosine similarity so that the process would be consistent with euclidean distance. In other words, with euclidean, we choose the k neighbors with the *smallest* distance values. By storing 1 - the cosine similarity, we can also choose the k neighbors with the *smallest* values.



## Analysis of Results
In the main.py file, we demoed three different versions of the KNN.
1. KNN with auto-k selection using euclidean (chosen k=3)
2. KNN with auto-k selection using cosim (chosen k=3)
3. KNN with k set to 2 using euclidean
The second model gave the best overall accuracy (0.91) out the three, so we will analyze these results.



             ==========Classification Report==========
              precision    recall  f1-score   support

           0       0.89      0.89      0.89        18
           1       0.93      1.00      0.96        27
           2       0.86      1.00      0.93        19
           3       0.79      0.83      0.81        18
           4       0.92      0.96      0.94        25
           5       0.91      0.77      0.83        13
           6       0.86      0.92      0.89        13
           7       1.00      0.96      0.98        24
           8       0.95      0.86      0.90        21
           9       0.89      0.77      0.83        22

    accuracy                           0.91       200
   macro avg       0.90      0.90      0.90       200
weighted avg       0.91      0.91      0.90       200


![[confusion_matrix_KNN Auto-k with cosim.png]]

Overall, the f1-scores for each of the number images ranged from 0.81 to 0.98.

**Recall**
We witness some issues with recall for 5 and 9 (both had low recall of 0.77).
- This means that the ability to properly identify true 5's and 9's was difficult (high false negatives).
- Out of 13 real 5's in the test set, 2 of them were misclassified as a 3 and one as a 6
- Out of 22 9's in the test set, 5 of them were misclassified as either 1, 3, 4, or 6

- 1's and 2's had recall of 1. This means all the real 1's and 2's were predicted as 1's and 2's, respectively. This might make sense since 1's and 2's are very simple to draw as they single lines

**Precision**
The number with the lowest precision was 3 (precision=0.79).
- This means other numbers were predicted as 3s at a high rate
- Two real 5's were predicted to be 3s, and real 8 and real 9 were predicted as a 3

The number 7 had a perfect precision of 1, so no other number was predicted as a 7.



## Appendix

### Exploratory Data Analysis - PCA on Training Set
![[pca_eda.png]]


![[pca_eda_zoomed.png]]


75 principal components explain about 88% of the variability in the training set.