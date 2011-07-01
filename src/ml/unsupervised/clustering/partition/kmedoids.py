# -*- coding: utf-8 -*-

import numpy as np


def partitioning_around_medoids(k, distances):
    """Partitioning Around Medoids (PAM) algorithm for hard clustering

    This is a classical realization of the k-medoids algorithm. It's an
    iterative technique that forms k clusters from the data by minimizing the
    sum of the clusters within-distances. Each cluster is represented by a data
    instance, called medoid. The remaining data instances are associated to the
    clusters with the closest medoids. The general loop selects medoids such
    that the cost criterion decreases incrementally. This iteration finishes
    when the medoids converges.

    For a detailed description of this algorithm, see [1], [2].

    [1] S. Theodoridis, K. Koutroumbas; Pattern Recognition; Fourth Edition;
        Academic Press; 2009
    [2] http://en.wikipedia.org/wiki/K-medoids

    This function receives the number of clusters and a pairwise distance matrix
    computed from the data to be clustered.

    The outputs are the labels for the data, the medoids indices and cost
    estimatives for each cluster.
    """

    assert k > 0, "There must be at least one cluster."

    distances = np.asarray(distances)

    assert distances.ndim == 2 and distances.shape[0] == distances.shape[1] \
        and np.all(distances.T==distances), "Invalid distance matrix."
    assert k <= distances.shape[0], "There must be more data than clusters."

    labels = None
    medoids_idx = np.random.permutation(distances.shape[0])[0:k]
    while True:
        new_labels = distances[:, medoids_idx].argmin(axis=1)
        if np.all(labels==new_labels):
            break
        labels = new_labels
        medoids_idx = [distances[labels==i, :].sum(axis=0).argmin() \
            for i in xrange(k)]
    cost = [distances[labels==i, medoids_idx[i]].sum() for i in xrange(k)]
    return labels, medoids_idx, cost