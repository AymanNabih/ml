# -*- coding: utf-8 -*-

import numpy as np


def mode(data):
    """Returns a list of the modal values in the data and the respective
    count"""

    data = np.asarray(data).ravel()
    counts = {}
    for d in data:
        counts[d] = counts.get(d, 0) + 1
    max_count = max(counts.values())
    mode_list = [x for x in counts if counts[x] == max_count]
    return mode_list, max_count