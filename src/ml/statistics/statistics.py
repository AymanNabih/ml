# -*- coding: utf-8 -*-


def bincount(data, weights=None):
    """Measures the weighted frequency of the data elements.

    Parameters
    ----------
    data : 1D-array
        The data.
    weights : 1D-array, optional
        Weights for the data. It must be an array the same size as the data.
        weights[i] measures the contribution of data[i]. When not specified,
        it's assumed that weights[i] == 1 for all i.

    Returns
    -------
    counts : dict
        A mapping between a data element and its frequency.

    Examples
    --------

    Without weights:

    >>> bincount([1, 3, 4, 4, 5, 4, 3, 2, 3, 4, 2, 2, 3])
    ... {1: 1, 2: 3, 3: 4, 4: 4, 5: 1}

    With weights:

    >>> bincount([1, 3, 4, 4], [0.5, 2, 1, 1])
    ... {1: 0.5, 3: 2, 4: 2}
    >>> bincount([1, 3, 4, 4, 4], [0.5, 2, 1, 0.5, 1])
    ... {1: 0.5, 3: 2, 4: 2.5}
    """

    if weights is None:
        weights = [1] * len(data)
    counts = {}
    for i in xrange(len(data)):
        counts[data[i]] = counts.get(data[i], 0) + weights[i]
    return counts

def mode(data, weights=None):
    """Returns a list of the modal values in the data and the respective
    count.

    Parameters
    ----------
    data : 1D-array
        The data.
    weights : 1D-array, optional
        Weights for the data. It must be an array the same size as the data.
        weights[i] measures the contribution of data[i]. When not specified,
        it's assumed that weights[i] == 1 for all i.

    Returns
    -------
    mode_list : 1D-array
        A list with the modal values.
    max_count : numeric
        The weighted frequency associated with the modal values.

    See Also
    --------
    ml.statistics.statistics.bincount

    Examples
    --------

    Without weights:

    >>> mode([1, 3, 4, 4, 5, 4, 3, 2, 3, 4, 2, 2, 3])
    ... ([3, 4], 4)

    With weights:

    >>> mode([1, 3, 4, 4], [0.5, 2, 1, 1])
    ... ([3, 4], 2)
    >>> mode([1, 3, 4, 4, 4], [0.5, 2, 1, 0.5, 1])
    ... ([4], 2.5)
    """

    counts = bincount(data, weights)
    max_count = max(counts.values())
    mode_list = list(set(k for k, v in counts.items() if v==max_count))
    return mode_list, max_count