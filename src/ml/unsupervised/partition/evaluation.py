# -*- coding: utf-8 -*-

import numpy as np

from scipy.spatial.distance import cdist


# TODO organize documentation using reST


def map_clusters_to_classes_lbls(clusters_lbls, classes_data, classes_lbls):
    """Map clusters to classes labels.

    This function uses a very simple heuristic to map clusters to classes
    labels. Before explaining the mechanics of it, it's important to understand
    the motivation behind.

    By definition, clustering procedures are suitable when there's no prior
    knowledge about the underlying structure of a bunch of data, i.e.,
    unsupervised situations. Consequently, the labels outputted by a clustering
    method cannot be interpreted by its own, e.g., if k-means runs twice there's
    no guarantee that label 0 in the first execution corresponds to label 0 in
    the second one.

    Despite clustering being a typical unsupervised procedure, sometimes is
    desirable to apply it to supervised data in order to evaluate the
    performance of a specific algorithm. However, because the clusters labels
    are not stable, it's not possible to compare them to classes labels
    directly. Thus, it's necessary to compute a mapping between clusters an
    classes labels before comparison.

    This function employs a simple idea. It's supposed that the cluster
    corresponding to a given class must have mean closer to the class mean,
    where cluster/class mean is the mean of the elements in the cluster/class.
    Based on this assumption, the clusters are mapped to the "corresponding"
    classes. The notion of "closer" is formalized using a distance function (in
    this case, the Euclidean).
    """

    clusters_lbls = np.asarray(clusters_lbls).ravel()
    classes_lbls = np.asarray(classes_lbls).ravel()

    assert len(clusters_lbls) == len(classes_lbls), "Classes and clusters " \
        "labels must be vectors of the same dimensionality."
    assert clusters_lbls.dtype == classes_lbls.dtype, "Classes and clusters " \
        "labels must be vectors of the same type."

    classes_data = np.asarray(classes_data)

    assert classes_data.ndim == 2, "The classes data must be a MxN matrix " \
        "formed by M observations of N features."
    assert classes_data.shape[0] == len(classes_lbls), "There's no " \
        "correspondance between classes data and labels."

    clusters = np.unique(clusters_lbls)
    classes = np.unique(classes_lbls)

    assert len(clusters) == len(classes), "There must be the same number of " \
        "classes and clusters."

    classes_means = [classes_data[classes_lbls==i].mean(axis=0) \
        for i in classes]
    clusters_means = [classes_data[clusters_lbls==i].mean(axis=0) \
        for i in clusters]
    new_clusters_lbls = clusters_lbls.copy()
    classes = classes[cdist(classes_means, clusters_means).argmin(axis=0)]
    for i in xrange(len(clusters)):
        new_clusters_lbls[clusters_lbls==clusters[i]] = classes[i]
    return new_clusters_lbls

def clustering_hit_rate(clusters_lbls, classes_lbls):
    """Evaluates the performance of a clustering (overall/per cluster hit rate).

    It's assumed there's a correspondance between clusters and classes labels.
    """

    clusters_lbls = np.asarray(clusters_lbls).ravel()
    classes_lbls = np.asarray(classes_lbls).ravel()

    assert len(clusters_lbls) == len(classes_lbls), "Classes and clusters " \
        "labels must be vectors of the same dimensionality."
    assert clusters_lbls.dtype == classes_lbls.dtype, "Classes and clusters " \
        "labels must be vectors of the same type."

    overall_hit_rate = np.count_nonzero(clusters_lbls==classes_lbls) / \
        float(len(clusters_lbls))
    clusters_hit_rates = []
    for c in np.unique(classes_lbls):
        n_matches = np.count_nonzero(clusters_lbls[classes_lbls==c]==c)
        n_class = np.count_nonzero(classes_lbls==c)
        clusters_hit_rates.append(float(n_matches) / n_class)
    return overall_hit_rate, clusters_hit_rates