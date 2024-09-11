import numpy as np
from distances import euclidean, cosim


def euclidean_dist(v1, v2):
    dist = np.sqrt(np.sum((v1 - v2) ** 2))
    return dist


def cosine_sim(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    dist = dot_product / (magnitude_v1 * magnitude_v2)
    return float(dist)


if __name__ == "__main__":

    v1 = np.random.rand(7)
    v2 = np.random.rand(7)

    l_v1 = v1.tolist()
    l_v2 = v2.tolist()

    d1 = euclidean_dist(v1, v2)

    d2 = euclidean(l_v1, l_v2)

    if d1 == d2:
        print("Euclidean test case passed!")

    c1 = cosine_sim(v1, v2)
    c2 = cosim(l_v1, l_v2)
    print(c1, c2)
    print(type(c1), type(c2))
    if c1 == c2:
        print("Cosine test cases passed!")
