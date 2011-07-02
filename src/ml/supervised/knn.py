# -*- coding: utf-8 -*-

import numpy as np

from ml.statistics.statistics import mode

from scipy.spatial.distance import cdist


def knn_classifier_gen(data, labels, k=1, weighted=False,
                       dist_metric='euclidean', thresh=np.inf, **dist_kwargs):
    data = np.copy(data)

    assert data.ndim == 2, "The data must be a MxN matrix formed by M " \
        "observations of N features."

    labels = np.copy(labels).ravel()

    assert data.shape[0] == len(labels), "There's no correspondance between " \
        "the data and the labels."

    d = data.shape[1]

    def classifier(x):
        x = np.asarray(x).ravel()

        assert len(x) == d, "Incorrect dimensionality. The input data must " \
            "be %d-dimensional." % d

        dists = cdist([x], data, metric=dist_metric, **dist_kwargs).squeeze()
        sorted_dists_idx = dists.argsort()[0:k]
        weights = 1 / (dists[sorted_dists_idx] ** 2) if weighted else None
        return mode(labels[sorted_dists_idx], weights)[0][0]

    return classifier