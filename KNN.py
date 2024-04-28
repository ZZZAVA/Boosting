import numpy as np
from collections import Counter
import csv
import random
import copy
from sklearn.tree import DecisionTreeClassifier


def distance(instance1, instance2):
    s = 0
    for i in range(len(instance1) - 1):
        s += 1 * abs(instance1[i] - instance2[i])
    return s


def get_distances(train_main, test_main):
    train = copy.deepcopy(train_main)
    test = copy.deepcopy(test_main)
    for i in range(len(test_main)):
        test[i].pop(-1)
    for i in range(len(train_main)):
        train[i].pop(-1)
    distances = {}
    test_ind = 0
    for i in test:
        distances[test_ind] = []
        train_ind = 0
        for j in train:
            distances[test_ind].append([train_ind, distance(i, j)])
            train_ind += 1
        test_ind += 1
    return distances


def get_neighbors(train, train_labels, test_num, k, distances):
    distances[test_num].sort(key=lambda x: x[1])
    neighbors = copy.deepcopy(distances[test_num][:k])
    for i in range(k):
        neighbors[i].append(train_labels[neighbors[i][0]])
        neighbors[i].append(train[neighbors[i][0]][-1])
    return neighbors


def voting(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += neighbor[3]
    return class_counter.most_common(1)[0][0]


def set_weight(right_label, wrong_label, train, train_labels, neighbors):
    for i in neighbors:
        j = i[0]
        if train_labels[j] == right_label:
            train[j][-1] *= 2
        elif train_labels[j] == wrong_label:
            train[j][-1] /= 2


def knn(main_train, main_train_labels):
    for ix in range(len(main_train)):
        k = 6
        train_labels = copy.deepcopy(main_train_labels)
        test = [main_train[ix]]
        test_labels = [train_labels[ix]]
        if ix == 0:
            train = main_train[1:]
        elif ix == len(main_train):
            train = main_train[:ix]
        else:
            train = main_train[:ix] + main_train[ix + 1:]
        d = get_distances(train, test)
        votes = []
        for test_ind in range(len(test)):
            neighbors = get_neighbors(train, train_labels, test_ind, k, d)
            votes.append(voting(neighbors))
        errors = {}
        for i in range(len(votes)):
            if votes[i] != test_labels[i]:
                errors[i] = (test_labels[i], votes[i])

        for i in errors:
            set_weight(errors[i][0], errors[i][1], train, train_labels,
                       get_neighbors(train, train_labels, i, k + 1, d))

    return main_train


def main(dataset, dataset_name):
    rows = len(dataset)
    dataset[0].insert(-1, "weight")
    for i in range(rows):
        dataset[i].insert(-1, 1)
    dataset.pop(0)
    random.shuffle(dataset)
    train_size = int(0.7 * rows)
    train_main = dataset[:train_size]
    test_main = dataset[train_size:]
    main_train_labels = [row[-1] for row in train_main]
    main_test_labels = [row[-1] for row in test_main]

    train = knn(train_main, main_train_labels)
    k = 8
    test = test_main
    train_labels = copy.deepcopy(main_train_labels)
    test_labels = copy.deepcopy(main_test_labels)
    d = get_distances(train, test)
    votes = []
    for test_ind in range(len(test)):
        neighbors = get_neighbors(train, train_labels, test_ind, k, d)
        votes.append(voting(neighbors))
    errors = {}
    for i in range(len(votes)):
        if votes[i] != test_labels[i]:
            errors[i] = test_labels[i]

    print(dataset_name + " knn err:", str(float(len(errors)) / len(test)))


def KNN_Gen(filenam_veh, filename_glass):
    main(filenam_veh, 'vehicle')
    main(filename_glass, 'glass')
