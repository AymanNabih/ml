# -*- coding: utf-8 -*-

import numpy as np


def partitioning_around_medoids(k, distances):
    """Partitioning Around Medoids (PAM) algorithm for hard clustering.

    This is a classical realization of the k-medoids algorithm. It's an
    iterative technique that forms k clusters from the data by minimizing the
    sum of the clusters within-distances. In each iteration, data instances
    called medoids are chosen to represent the clusters. The remaining data
    instances are associated to the clusters with the closest medoids. This
    process is repeated until the medoids converge.

    For a detailed description of this algorithm, see [Theodoridis2009]_,
    [Wikipedia]_.

    Parameters
    ----------
    k : integer
        The number of clusters. Must be > 0 and <= # data instances.
    distances : 2D-array
        Pairwise distance matrix computed from the data to be clustered. Of
        course, by definition, it must be a symmetric matrix.

    Returns
    -------
    labels : 1D-array
        Labels for the data.
    medoids_idx : 1D-array
        The indices of the clusters centers.
    cost : 1D-array
        The cost associated with each cluster.

    Raises
    ------
    ValueError
        If an empty cluster is created.

    References
    ----------
    .. [Theodoridis2009] S. Theodoridis, K. Koutroumbas; Pattern Recognition;
                         Fourth Edition; Academic Press; 2009
    .. [Wikipedia] http://en.wikipedia.org/wiki/K-medoids
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
        if len(np.unique(medoids_idx)) != k:
            raise ValueError("Empty cluster created.")
    cost = [distances[labels==i, medoids_idx[i]].sum() for i in xrange(k)]
    return labels, medoids_idx, cost