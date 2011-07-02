# -*- coding: utf-8 -*-

import numpy as np


def stratified_k_fold(labels, k):
    """An iterator providing *test* data masks for stratified k-fold validation.

    Each iteration returns a mask that can be used to instantiate both training
    and test data for validation. Check the examples.

    For detailed information about stratified k-fold validation, see
    [Wikipedia]_.

    Parameters
    ----------
    labels : 1D-array
        The data labels.
    k : integer
        The number of folds. It must be > 1 and <= the cardinality of the less
        frequent class. Otherwise, it'd not be possible to guarantee
        stratification.

    Returns
    -------
    out : iterator
        An iterator providing *test* data masks for each fold.

    Examples
    --------

    The following snippet instatiates training and test data k times according
    to a stratified k-fold validation scheme:

    >>> for test_data_mask in stratified_k_fold(labels, k):
    ...     training_data = dataset[~test_data_mask]
    ...     test_data = dataset[test_data_mask]

    References
    ----------
    .. [Wikipedia] http://en.wikipedia.org/wiki/Cross-validation_(statistics)
    """

    assert k > 1, "There must be at least two folds."

    labels = np.asarray(labels).ravel()

    if k > min(np.count_nonzero(labels==c) for c in np.unique(labels)):
        raise ValueError("'k' must be <= the cardinality of the less frequent "
            "class. Otherwise, it'd not be possible to guarantee "
            "stratification.")
    sorted_labels_idx = labels.argsort()
    for i in xrange(k):
        test_filter = np.zeros(len(labels), dtype=np.bool)
        test_filter[sorted_labels_idx[i::k]] = True
        yield test_filter

def classifier_hit_rate(test_data, test_lbls, classifier):
    """Evaluates the performance of a *single feature space* classifier (overall
    hit rate).

    Parameters
    ----------
    test_data : 2D-array
        The test data. This must be a MxN matrix formed by M observations of N
        features.
    test_lbls : 1D-array
        The test labels. This must be an array the same size as the test data
        rows.
    classifier : function
        The classifier function. This is a function f : V -> L where V is the
        feature space and L is the class space.

    Returns
    -------
    out : float
        The overall hit rate of the classifier over the test set.
    """

    test_data = np.asarray(test_data)

    assert test_data.ndim == 2, "The test data must be a MxN matrix formed " \
        "by M observations of N features."

    test_lbls = np.asarray(test_lbls).ravel()

    assert test_data.shape[0] == len(test_lbls), "There's no correspondance " \
        "between the data and the labels."

    n_matches = np.count_nonzero([classifier(t) for t in test_data]==test_lbls)

    return n_matches / float(test_data.shape[0])