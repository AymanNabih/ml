# -*- coding: utf-8 -*-

# TODO reimplement all this stuff in an elegant way =S

import numpy as np

from functools import partial

from ml.statistics.statistics import mode


def gen_product_rule(priors, classifiers_discriminants, fun, same_features_space):
    n_classifiers = classifiers_discriminants.shape[0]

    def not_same_features_space_rule(i, x):
        discriminants = classifiers_discriminants[:, i]
        evaluation = [discriminants[i](x[i]) for i in xrange(len(discriminants))]
        return fun(evaluation) * priors[i] ** -(n_classifiers - 1)

    def same_features_space_rule(i, x):
        return fun([d(x) for d in classifiers_discriminants[:, i]]) * \
            priors[i] ** -(n_classifiers - 1)

    return same_features_space_rule if same_features_space else \
        not_same_features_space_rule

def product_rule(priors, classifiers_discriminants, same_feature_space):
    return gen_product_rule(priors, classifiers_discriminants, np.prod, \
        same_feature_space)

def min_rule(priors, classifiers_discriminants, same_feature_space):
    return gen_product_rule(priors, classifiers_discriminants, np.min, \
        same_feature_space)

def gen_sum_rule(priors, classifiers_discriminants, fun, same_feature_space):
    n_classifiers, n_classes = classifiers_discriminants.shape

    def p1(x):
        return np.asarray([d(x) for d in classifiers_discriminants.ravel()]). \
            reshape(classifiers_discriminants.shape)

    def p2(x):
        ret = []
        for i in xrange(classifiers_discriminants.shape[1]):
            discriminants = classifiers_discriminants[:, i]
            ret.append([discriminants[k](x[k]) for k in xrange(len(discriminants))])
        ret = np.asarray(ret)
        ret = ret.T
        return ret

    def f(i, x):
        if same_feature_space:
            p = p1
        else:
            p = p2
        p_x = p(x)
        posterior = p_x / np.sum(p_x, axis=1).reshape(p_x.shape[0], 1)
        return fun(posterior[:, i]) + priors[i] * (1 - n_classifiers)

    return f

def sum_rule(priors, classifiers_discriminants, same_feature_space):
    return gen_sum_rule(priors, classifiers_discriminants, np.sum, same_feature_space)

def max_rule(priors, classifiers_discriminants, same_feature_space):
    fun = lambda x: classifiers_discriminants.shape[0] * np.max(x)
    return gen_sum_rule(priors, classifiers_discriminants, fun, same_feature_space)

def median_rule(priors, classifiers_discriminants, same_feature_space):
    return gen_sum_rule(priors, classifiers_discriminants, np.median, same_feature_space)

def combined_discriminants_classifier(classes, priors, \
        classifiers_discriminants, comb_rule, same_feature_space=True):
    classes = np.asarray(classes)
    priors = np.asarray(priors)
    classifiers_discriminants = np.asarray(classifiers_discriminants)

    discriminant = comb_rule(priors, classifiers_discriminants, \
        same_feature_space)

    classifier = lambda discriminants, x: classes[np.argmax([d(x) \
       for d in discriminants])]

    discriminants = [partial(discriminant, i) for i in xrange(classes.size)]
    return partial(classifier, discriminants), discriminants

def majority_vote_classifier(classifiers, same_feature_space=True):
    assert len(classifiers) % 2 != 0
    if same_feature_space:
        classifier = lambda x: mode([c(x) for c in classifiers])[0][0]
    else:
        classifier = lambda x: mode([classifiers[i](x[i]) \
            for i in xrange(len(classifiers))])[0][0]
    return classifier