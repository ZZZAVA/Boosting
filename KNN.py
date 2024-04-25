import random

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import *
from sklearn.neighbors.base import _check_weights


def KNN_Gen(X_veh, y_veh, X_glass, y_glass):
    acc = 0;
    test_count = 10
    dependence_test = Series(index=range(1, 35, 1), dtype=float)
    # k_res = [ ]
    for n in range(1, 35, 1):
        mean_accuracy_test = 0.
        for _ in range(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X_veh, y_veh, random_state=random.randint(0, 1000),
                                                                train_size=0.7)
            clf = AdaBoostClassifier(base_estimator=KNN(n_neighbors=n), algorithm='SAMME')
            clf.fit(X_train, np.array(y_train).ravel())
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
            dependence_test[n] = mean_accuracy_test
            acc =mean_accuracy_test

    print("Vehicle err(n):", 1-acc)
    plt.style.use('seaborn')
    plt.plot(dependence_test, label='test', marker='.', markersize=1)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.title('Vehicle')
    plt.show()
    dependence_test = Series(index=range(1, 301, 10), dtype=float)
    # k_res = [ ]
    for n in range(1, 301, 10):
        mean_accuracy_test = 0.
        for _ in range(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X_veh, y_veh, random_state=random.randint(0, 1000),
                                                                train_size=0.7)
            clf = AdaBoostClassifier(base_estimator=KNN(n_neighbors=5), n_estimators=n, algorithm='SAMME')
            clf.fit(X_train, np.array(y_train).ravel())
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
            dependence_test[n] = mean_accuracy_test
            acc =mean_accuracy_test

    print("Vehicle err(est):", 1-acc)
    plt.style.use('seaborn')
    plt.plot(dependence_test, label='test', marker='.', markersize=1)
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')
    plt.title('Vehicle')
    plt.show()
    test_count = 10

    dependence_test = Series(index=range(1, 35, 1), dtype=float)
    for n in range(1, 35, 1):
        mean_accuracy_test = 0.
        for _ in range(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X_glass, y_glass, random_state=random.randint(0, 1000),
                                                                train_size=0.7)
            clf = AdaBoostClassifier(base_estimator=KNN(n_neighbors=n), algorithm='SAMME')
            clf.fit(X_train, np.array(y_train).ravel())
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
            dependence_test[n] = mean_accuracy_test
            acc =mean_accuracy_test

    print("Glass err(n):", 1-acc)
    plt.style.use('seaborn')
    plt.plot(dependence_test, label='test', marker='.', markersize=1)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.title('Glass')
    plt.show()


    for n in range(1, 301, 10):
        mean_accuracy_test = 0.
        for _ in range(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X_glass, y_glass, random_state=random.randint(0, 1000),
                                                                train_size=0.7)
            clf = AdaBoostClassifier(base_estimator=KNN(n_neighbors=5), n_estimators=n, algorithm='SAMME')
            clf.fit(X_train, np.array(y_train).ravel())
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
            dependence_test[n] = mean_accuracy_test
            acc =mean_accuracy_test

    print("Glass err(est):", 1-acc)
    plt.style.use('seaborn')
    plt.plot(dependence_test, label='test', marker='.', markersize=1)
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')
    plt.title('Glass')
    plt.show()

class KNN(KNeighborsClassifier):
    def __init__(self, n_neighbors, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=None, **kwargs):
        super(KNeighborsClassifier, self).__init__(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size,
                                                   metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs,
                                                   **kwargs)

        self.weights = _check_weights(weights)

    def fit(self, X, y, sample_weight):
        return super(KNN, self).fit(X, y)

