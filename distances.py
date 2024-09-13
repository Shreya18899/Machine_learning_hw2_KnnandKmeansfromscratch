# File containing the distance metrics
# Euclidean and cosim distances computed for kmeans and knn

def euclidean(a, b):
    """
    return: Euclidean distance between vectors a and b
    """
    return sum((x-y)**2 for x, y in zip(a, b))**0.5


def cosim(a, b):
    """
    return: Cosine Similarity between vectors a and b
    """
    dot_product = sum(vect1 * vect2 for vect1, vect2 in zip(a, b))
    magnitude_v1 = sum(vect1 ** 2 for vect1 in a) ** 0.5
    magnitude_v2 = sum(vect2 ** 2 for vect2 in b) ** 0.5
    # Handle for zero division
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0

    dist = dot_product / (magnitude_v1 * magnitude_v2)
    return dist
