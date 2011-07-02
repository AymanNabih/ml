# -*- coding: utf-8 -*-

import numpy as np

from functools import partial

from ml.statistics.distributions import standard_normal_pdf


def normal_kernel(x):
    """Multivariate normal kernel."""

    return standard_normal_pdf(x)

def box_kernel(x):
    """Multivariate box kernel."""

    return np.all(np.abs(np.asarray(x).ravel())<0.5)

def kde(x, data, h, kernel=normal_kernel):
    """Multivariate Kernel Density Estimation algorithm (also known as
    Parzen-Rosenblatt Window algorithm).

    This algorithm employs a non-parametric method to estimate the probability
    density of a multivariate random variable. The general idea is to
    approximate the density by interpolation over a small fixed size (bandwidth)
    window using a finite number of reference samples. The interpolation is
    controlled by a special function called kernel, which must be >= 0 and
    integrate to one over the domain.

    For more information about this algorithm, see [Duda2000]_.

    *PS: This implementation uses an isotropic bandwidth matrix.*

    Parameters
    ----------
    x : 1D-array
        The test sample. This must be a vector with the same dimensionality of
        the reference samples.
    data : 2D-array
        The reference samples. This must be a MxN matrix formed by M
        observations of N features.
    h : float
        The size of the window (bandwidth). Must be > 0.
    kernel : function, optional
        The interpolation kernel. It's normal by default.

    Returns
    -------
    out : float
        Estimated density evaluated at the test sample.

    References
    ----------
    .. [Duda2000] R. O. Duda, P. E. Hart, D. G. Stork; Pattern Classification;
                  Second Edition; Wiley-Interscience; 2000
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
    """Generator of Bayesian classifier based on Kernel Density Estimation.

    This algorithm generates a Bayesian classifier by approximating p(x|class)
    using Kernel Density Estimation.

    For more information about Kernel Density Estimation and Bayesian
    classification, see [Duda2000]_.

    Parameters
    ----------
    data : 2D-array
        The training data. This must be a MxN matrix formed by M observations of
        N features.
    labels : 1D-array
        The training labels. This must be an array the same size as the training
        data rows.
    h : float
        The size of the window (bandwidth). Must be > 0.
    kernel : function, optional
        The interpolation kernel. It's normal by default.
    priors : 1D-array, optional
        Priors estimatives for each class. This must be an array the same size
        as unique(labels) and sum to one (tolerance of 1e-5). It's calculated
        from the data by default.

    Returns
    -------
    classifier : function
        The classifier function. This is a function f : V -> L where V is the
        feature space and L is the class space.
    discriminants : 1D-array of functions
        A list of the Bayesian discriminants (non-normalized posteriors) for
        each class. The Bayesian discriminant for the c class is a function
        g_c(x)=p(x|c)*p(c). So, it's possible to get the posterior by
        p(c|x)=g_c(x)/(sum_c(g_c(x)).
    priors : 1D-array
        The echo of the priors if informed or a list of estimated values from
        the data.

    See Also
    --------
    ml.supervised.kde.kde
        The Kernel Density Estimation algorithm.
    ml.supervised.gda.gda_classifier_gen
        Similar function using Gaussian Discriminant Analysis.

    References
    ----------
    .. [Duda2000] R. O. Duda, P. E. Hart, D. G. Stork; Pattern Classification;
                  Second Edition; Wiley-Interscience; 2000
    """

    assert h > 0, "The window size must be greater than 0."

    data = np.copy(data)

    assert data.ndim == 2, "The data must be a MxN matrix formed by M " \
        "observations of N features."

    n, d = data.shape
    labels = np.copy(labels).ravel()

    assert n == len(labels), "There's no correspondance between " \
        "the data and the labels."

    classes = np.unique(labels)
    if priors is None:
        priors = [float(np.count_nonzero(labels==c))/len(labels) \
            for c in classes]
    else:
        priors = np.copy(priors)
        assert len(priors) == len(classes), "There's no correspondance " \
            "between the specified priors and the detected classes."

    assert abs(sum(priors) - 1.0) < 1e-5, "The priors must sum to one."

    discriminant = lambda i, x: kde(x, data[labels==classes[i]], h, kernel) * \
        priors[i]
    discriminants = [partial(discriminant, i) for i in xrange(len(classes))]

    def classifier(x):
        x = np.asarray(x).ravel()

        assert d == len(x), "Incorrect dimensionality. The input data must " \
            "be %d-dimensional." % d

        return classes[np.argmax([g(x) for g in discriminants])]

    return classifier, discriminants, priors