**Q5. Prepare a soft k-means classifier using the guideline provided for Question #3 above.**

**Design choices** :-
1. Similar to the KMeans in step 3, we assume an MNIST dataset, we have taken a default of 10 clusters.
2. Hyperparameters:
a. max_iter = ensures that the algorithm converges. 
b. Beta = A second hyperparameter of beta (temperature )is added for which we tune the model to a range of values and test the classification accuracies. A value of beta generally lower than 1 gives better performance when assessed by accuracies. 
3. Tolerance = We have set a value of tolerance = 1e-4, that determines the stopping condition for the algorithm based on how much the cluster centroids changes between iterations.
4. Label assignment = Though soft kmeans uses a probabilistic distribution to fit the labels, the prediction part assigns hard labels and this is common practise which we have taken into consideration while assigning the clusters and testing it out.
5. Dimensionality reduction = No dimensionality reduction was performed for this data as it dropped the performance of the algorithm. The scaling is done in the same way as Kmeans using a minmax scaling technique.
6. Perform validation = The classification accuracies are obtained in the same way as step 3.

**Analysis**
1. Distance metric - cosine similarity works as a much more effective metric for the soft kmeans
2. While the precision is higher for certain clusters, the low values of precision for certain clusters is bringing down the overall accuracy.
3. Beta - values of beta much lesser than 1 are effective in clustering.
4. Higher recall values for soft k means is observed. When we say "higher recall", it means that a larger proportion of the actual positive cases were identified correctly.
5. Inconsistency is observed in the classifications which signifies room for improvement since no definitive clusters seem to be underperforming.
