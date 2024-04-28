import random

import matplotlib.pyplot as plt
from pandas import Series
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def Tree_Gen(X_veh, y_veh, X_glass, y_glass):
    test_count = 10
    acc = 0
    dependence_test = Series(index=range(1, 301, 10), dtype=float)
    for n in range(1, 301, 10):
        mean_accuracy_test = 0.
        for _ in range(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X_veh, y_veh, random_state=random.randint(0, 1000), train_size=0.7)
            clf = DecisionTreeClassifier(max_depth = 1, random_state = 0)
            clf.fit(X_train, y_train)
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
            dependence_test[n] = mean_accuracy_test
            acc =mean_accuracy_test

    print("vehicle tree err:", 1-acc)

    dependence_test = Series(index=range(1, 301, 10), dtype=float)
    for n in range(1, 301, 10):
        mean_accuracy_test = 0.
        for _ in range(test_count):
            X_train, X_test, y_train, y_test = train_test_split(X_glass, y_glass, random_state=random.randint(0, 1000),
                                                                train_size=0.7)
            clf = DecisionTreeClassifier(max_depth = 1, random_state = 0)
            clf.fit(X_train, y_train)
            mean_accuracy_test += clf.score(X_test, y_test) / test_count
            dependence_test[n] = mean_accuracy_test
            acc =mean_accuracy_test

    print("glass tree err:", 1-acc)

