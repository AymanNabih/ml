# -*- coding: utf-8 -*-

import numpy as np

from functools import partial

from ml.statistics.distributions import normal_pdf


# IMPROVEMENT this and the KDE classifier shares the same structure. It's a
#             discriminant classifier. A possible improvement is to refactor it
#             creating a generic discriminant_classifier_gen which receives a
#             discriminants_gen function. It seems a good design, but, for now,
#             it's a overkill.


def gda_classifier_gen(data, labels, priors=None, use_pooled_sigma=False):
    """Generator of Bayesian classifier based on Gaussian Discriminant Analysis.

    This algorithm generates a Bayesian classifier assuming the data for each
    class are distributed according to multivariate Gaussians. Given this
    assumption, the parameters of the densities are estimated using a Maximum
    Likelihood procedure.

    This approach is quite reasonable when each class can be represented by just
    one vector corrupted by many components of noise. In this case, the Gaussian
    assumption is self-evident given the Central Limit Theorem.

    For more information about Gaussian Discriminant Analysis and Bayesian
    classification, see [Duda2000]_, [CS229]_.

    Parameters
    ----------
    data : 2D-array
        The training data. This must be a MxN matrix formed by M observations of
        N features.
    labels : 1D-array
        The training labels. This must be an array the same size as the training
        data rows.
    use_pooled_sigma : boolean, optional
        Flag indicating if the multivariate Gaussians should use a pooled sigma.
        It's False by default.
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
        The echo of the priors if informed or a list of estimated priors from
        the data.

    See Also
    --------
    ml.supervised.kde.kde_classifier_gen
        Similar function using Kernel Density Estimation.

    References
    ----------
    .. [Duda2000] R. O. Duda, P. E. Hart, D. G. Stork; Pattern Classification;
                  Second Edition; Wiley-Interscience; 2000
    .. [CS229] http://www.stanford.edu/class/cs229/notes/cs229-notes2.pdf
    """

    data = np.asarray(data)

    assert data.ndim == 2, "The data must be a MxN matrix formed by M " \
        "observations of N features."

    n, d = data.shape
    labels = np.asarray(labels).ravel()

    assert n == len(labels), "There's no correspondance between " \
        "the data and the labels."

    classes = np.unique(labels)
    n_classes, mu, sigma = [], [], []
    for c in classes:
        n_classes.append(np.count_nonzero(labels==c))
        mu.append(data[labels==c].mean(axis=0))
        sigma.append(np.cov(data[labels==c], rowvar=0))

    if priors is None:
        priors = [float(n)/len(labels) for n in n_classes]
    else:
        priors = np.copy(priors)
        assert len(priors) == len(classes), "There's no correspondance " \
            "between the specified priors and the detected classes."

    assert abs(sum(priors) - 1.0) < 1e-5, "The priors must sum to one."

    if use_pooled_sigma:
        pooled_sigma = np.sum([(n_c-1.0)/(n-len(classes))*s  \
            for n_c, s in zip(n_classes, sigma)], axis=0)
        pdfs = [normal_pdf(mu[i], pooled_sigma) for i in xrange(len(classes))]
    else:
        pdfs = [normal_pdf(mu[i], sigma[i]) for i in xrange(len(classes))]

    discriminant = lambda i, x: pdfs[i](x) * priors[i]
    discriminants = [partial(discriminant, i) for i in xrange(len(classes))]

    def classifier(x):
        x = np.asarray(x).ravel()

        assert d == len(x), "Incorrect dimensionality. The input data must " \
            "be %d-dimensional." % d

        return classes[np.argmax([g(x) for g in discriminants])]

    return classifier, discriminants, priors