# -*- coding: utf-8 -*-

import numpy as np


def stratified_k_fold(labels, k):
    """An iterator providing data masks for stratified k-fold validation

    Each iteration returns a mask that can be used to instantiate both training
    and test data for validation. For example, the following snippet
    instatiates training and test data k times according to a stratified k-fold
    validation scheme.

    >>> for test_data_mask in stratified_k_fold(labels, k):
    ...     training_data = dataset[~test_data_mask]
    ...     test_data = dataset[test_data_mask]

    For detailed information about stratified k-fold validation, see [1].

    [1] http://en.wikipedia.org/wiki/Cross-validation_(statistics)
    """

    assert k > 1, "There must be at least two folds."

    labels = np.asarray(labels).ravel()

    if k > min(np.count_nonzero(labels==i) for i in np.unique(labels)):
        raise ValueError("'k' must be <= the cardinality of the less frequent "
            "class. Otherwise, it'd not be possible to guarantee "
            "stratification.")
    sorted_labels_idx = labels.argsort()
    for i in xrange(k):
        test_filter = np.zeros(len(labels), dtype=np.bool)
        test_filter[sorted_labels_idx[i::k]] = True
        yield  test_filter

def classifier_hit_rate(test_data, test_lbls, classifier):
    """Evaluates the performance of a single feature space classifier (overall
    hit rate)

    It receives the test data and labels, and a single feature space classifier.
    This classifier is a function g that can be used to classify patterns, e.g,
    g(test_data[0]).
    """

    test_lbls = np.asarray(test_lbls).ravel()
    test_data = np.asarray(test_data)

    assert test_data.ndim == 2, "The test data must be a MxN matrix formed " \
        "by M observations of N features."

    n_matches = np.count_nonzero([classifier(t) for t in test_data]==test_lbls)

    return n_matches / float(test_data.shape[0])