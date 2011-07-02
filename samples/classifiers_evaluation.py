# -*- coding: utf-8 -*-

"""
This is a simple sample showing how to evaluate the performance of different
classifiers.

Three synthetic classes are created according to bivariate normal distributions
with different means and covariances. The following classifiers are evaluated
using stratified k-fold:

    - GDA wihout pooled sigma
    - GDA with pooled sigma
    - KDE with h=0.1 and normal kernel
    - KNN with k=1, cityblock distance metric
    - Weighted-KNN with k=1, cityblock distance metric
"""

import numpy as np

from ml.supervised.evaluation import classifier_hit_rate, stratified_k_fold
from ml.supervised.gda import gda_classifier_gen
from ml.supervised.kde import kde_classifier_gen
from ml.supervised.knn import knn_classifier_gen


def class_data_gen(label, mu, sigma, n):
    data = np.random.multivariate_normal(mu, sigma, n)
    return np.column_stack((data, np.repeat(label, n)))

classes_cfg = [
    ([0, 0], [[4, 1.7], [1.7, 1]], 1500),
    ([0, 3], [[0.25, 0], [0, 0.25]], 1000),
    ([4, 3], [[4, -1.7], [-1.7, 1]], 500),
]

print "Classes configurations:"
for i in xrange(len(classes_cfg)):
    print "\tClass %d: Normal(mu=%s, sigma=%s); n = %d" % (i + 1, \
        classes_cfg[i][0], classes_cfg[i][1], classes_cfg[i][2])

print "Generating dataset..."
dataset = np.concatenate([(class_data_gen(lbl, *cfg)) for lbl, cfg in \
    enumerate(classes_cfg)])

classes_data = dataset[:, 0:-1]
classes_lbls = dataset[:, -1]

n_folds = 50
gda1_hr, gda2_hr, kde_hr, knn1_hr, knn2_hr = 0, 0, 0, 0, 0
print "Evaluating classifiers using stratified %d-fold validation..." % n_folds
for test_mask in stratified_k_fold(classes_lbls, n_folds):
    training_data = classes_data[~test_mask]
    training_lbls = classes_lbls[~test_mask]
    test_data = classes_data[test_mask]
    test_lbls = classes_lbls[test_mask]

    gda1_c = gda_classifier_gen(training_data, training_lbls)[0]
    gda1_hr += classifier_hit_rate(test_data, test_lbls, gda1_c)

    gda2_c = gda_classifier_gen(training_data, training_lbls, \
        use_pooled_sigma=True)[0]
    gda2_hr += classifier_hit_rate(test_data, test_lbls, gda2_c)

    kde_c = kde_classifier_gen(training_data, training_lbls, h=0.1)[0]
    kde_hr += classifier_hit_rate(test_data, test_lbls, kde_c)

    knn1_c = knn_classifier_gen(training_data, training_lbls)
    knn1_hr += classifier_hit_rate(test_data, test_lbls, knn1_c)

    knn2_c = knn_classifier_gen(training_data, training_lbls, weighted=True)
    knn2_hr += classifier_hit_rate(test_data, test_lbls, knn2_c)

print "Mean hit rates:"
print "\tGDA (use_pooled_sigma=False): %.4f" % (gda1_hr / n_folds)
print "\tGDA (use_pooled_sigma=True): %.4f" % (gda2_hr / n_folds)
print "\tKDE (h=0.1; kernel=normal): %.4f" % (kde_hr / n_folds)
print "\tKNN (k=1; weighted=False; dist=cityblock): %.4f" % (knn1_hr / n_folds)
print "\tKNN (k=1; weighted=True; dist=cityblock): %.4f" % (knn2_hr / n_folds)