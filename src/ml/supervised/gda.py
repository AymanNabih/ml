# -*- coding: utf-8 -*-

import numpy as np

from functools import partial

from ..statistics.distributions import normal_pdf


def gda_classifier_gen(data, labels, use_pooled_sigma=False, priors=None):
    """Generator of Bayesian classifier based on Gaussian Discriminant Analysis

    This algorithm generates a Bayesian classifier assuming the data for each
    class are distributed according to multivariate Gaussians. Given this
    assumption, the parameters of the densities are estimated using a Maximum
    Likelihood procedure.

    This approach is quite reasonable when each class can be represented by just
    one vector corrupted by many components of noise. In this case, the Gaussian
    assumption is self-evident given the Central Limit Theorem.

    For more information about Gaussian Discriminant Analysis and Bayesian
    classification, see [1], [2], [3].

    [1] R. O. Duda, P. E. Hart, D. G. Stork; Pattern Classification; Second
        Edition; Wiley-Interscience; 2000
    [2] http://www.stanford.edu/class/cs229/notes/cs229-notes2.pdf
    [3] http://en.wikipedia.org/wiki/Bayesian_inference

    This function receives the training data, the associated labels, a flag
    indicating if the multivariate Gaussians should use a pooled sigma (false by
    default) and a list of priors estimatives (it's calculated from the data by
    default).

    The outcome is another function that can be used to classify non-seen data,
    the Bayesian discriminants (non-normalized posteriors) for each class and
    the priors estimatives.

    PS: The Bayesian discriminant for the c class is a function
    g_c(x)=p(x|c)*p(c). So, it's possible to get the posterior by
    p(c|x)=g_c(x)/(sum_c(g_c(x)).
    """

    data = np.asarray(data)

    assert data.ndim == 2, "The data must be a MxN matrix formed by M " \
        "observations of N features."

    labels = np.asarray(labels).ravel()

    assert data.shape[0] == len(labels), "There's no correspondance between " \
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
        assert len(priors) == len(classes), "There's no correspondance " \
            "between the specified priors and the detected classes."

    assert abs(sum(priors) - 1.0) < np.finfo(float).eps, \
        "The priors must sum to one."

    if use_pooled_sigma:
        pooled_sigma = np.sum([(n-1.0)/(data.shape[0]-len(classes))*s \
            for n, s in zip(n_classes, sigma)], axis=0)
        pdfs = [normal_pdf(mu[i], pooled_sigma) for i in xrange(len(classes))]
    else:
        pdfs = [normal_pdf(mu[i], sigma[i]) for i in xrange(len(classes))]

    discriminant = lambda i, x: pdfs[i](x) * priors[i]
    discriminants = [partial(discriminant, i) for i in xrange(len(classes))]
    classifier = lambda x: classes[np.argmax([d(x) for d in discriminants])]

    return classifier, discriminants, priors
