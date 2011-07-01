# -*- coding: utf-8 -*-

import math as m
import numpy as np

from scipy.spatial.distance import mahalanobis


def standard_normal_pdf(x):
    """PDF for a multivariate standard normal distribution"""

    x = np.asarray(x).ravel()
    return ((2 * m.pi) ** (-0.5 * len(x))) * m.exp(-0.5*np.dot(x, x))

def normal_pdf(mu, sigma):
    """PDF for a multivariate normal distribution"""

    mu = np.asarray(mu).ravel()
    sigma = np.atleast_2d(sigma)
    d = len(mu)

    assert sigma.ndim == 2 and sigma.shape[0] == sigma.shape[1] and \
        np.all(sigma.T==sigma), "The covariance must be a symmetric matrix."
    assert sigma.shape[0] == d, "The dimensions of the mean and the " \
        "covariance matrix don't match."

    inv_sigma = np.linalg.inv(sigma)
    scale_factor = ((2 * m.pi) ** d * np.linalg.det(sigma)) ** -0.5

    def pdf(x):
        x = np.asarray(x).ravel()

        assert len(x) == d, "Incorrect dimensionality. The input data must " \
            "be %d-dimensional." % d

        return scale_factor * m.exp(-0.5*mahalanobis(x, mu, inv_sigma))

    return pdf