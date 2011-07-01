# -*- coding: utf-8 -*-

import numpy as np

from functools import partial

from ..statistics.distributions import standard_normal_pdf


def normal_kernel(x):
    """Multivariate normal kernel"""

    return standard_normal_pdf(x)

def box_kernel(x):
    """Multivariate box kernel"""

    return np.all(np.abs(np.asarray(x).ravel())<0.5)

def kde(x, data, h, kernel=normal_kernel):
    """Multivariate Kernel Density Estimation algorithm (also known as
    Parzen-Rosenblatt Window algorithm)

    This algorithm employs a non-parametric method to estimate the probability
    density of a multivariate random variable. The general idea is to
    approximate the density by interpolation over a small fixed sized window
    using a finite number of reference samples. The interpolation is controlled
    by a special function called kernel, which must be >= 0 and integrate to one
    over the domain.

    For more information about this algorithm, see [1], [2].

    [1] R. O. Duda, P. E. Hart, D. G. Stork; Pattern Classification; Second
        Edition; Wiley-Interscience; 2000
    [2] http://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation

    This function receives a test sample, reference samples, the size of the
    window (bandwidth) and the interpolation kernel (normal by default).

    The output is the estimated density evaluated at the test sample.

    PS: This implementation uses an isotropic bandwidth matrix.
    """

    data = np.asarray(data)

    assert data.ndim == 2, "The data must be a MxN matrix formed by M " \
        "observations of N features."

    x = np.asarray(x).ravel()
    n, d = data.shape

    assert len(x) == d, "The test sample must have the same dimensionality " \
        "of the reference samples."
    assert h > 0, "The window size must be greater than 0."

    return float(sum([kernel(v) for v in (x-data)/h])) / (n * h ** d)

def kde_classifier_gen(data, labels, h, kernel=normal_kernel, priors=None):
    """Generator of Bayesian classifier based on Kernel Density Estimation

    This algorithm generates a Bayesian classifier by approximating p(x|class)
    using Kernel Density Estimation.

    For more information about Kernel Density Estimation and Bayesian
    classification, see [1], [2], [3].

    [1] R. O. Duda, P. E. Hart, D. G. Stork; Pattern Classification; Second
        Edition; Wiley-Interscience; 2000
    [2] http://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation
    [3] http://en.wikipedia.org/wiki/Bayesian_inference

    This function receives the training data, the associated labels, the size of
    the window, the interpolation kernel (normal by default) and a list of
    priors estimatives (it's calculated from the data by default).
    
    The outcome is another function that can be used to classify non-seen data,
    the Bayesian discriminants (non-normalized posteriors) for each class and
    the priors estimatives.

    PS: The Bayesian discriminant for the c class is a function
    g_c(x)=p(x|c)*p(c). So, it's possible to get the posterior by
    p(c|x)=g_c(x)/(sum_c(g_c(x)).
    """

    assert h > 0, "The window size must be greater than 0."

    data = np.asarray(data)

    assert data.ndim == 2, "The training data must be a MxN matrix formed by " \
        "M observations of N features."

    labels = np.asarray(labels).ravel()

    assert data.shape[0] == len(labels), "There's no correspondance between " \
        "the data and the labels."

    classes = np.unique(labels)

    if priors is None:
        priors = [float(np.count_nonzero(labels==c))/len(labels) \
            for c in classes]
    else:
        assert len(priors) == len(classes), "There's no correspondance " \
            "between the specified priors and the detected classes."

    assert abs(sum(priors) - 1.0) < np.finfo(float).eps, \
        "The priors must sum to one."

    discriminant = lambda i, x: kde(x, data[labels==classes[i]], h, kernel) * \
        priors[i]
    discriminants = [partial(discriminant, i) for i in xrange(len(classes))]
    classifier = lambda x: classes[np.argmax([d(x) for d in discriminants])]

    return classifier, discriminants, priors
