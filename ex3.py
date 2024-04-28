import csv

import pandas as pd

from Tree import Tree_Gen
from KNN import KNN_Gen

data_veh = pd.read_csv('Vehicle.csv')
y_veh = data_veh['Class']
X_veh = data_veh.loc[:, 'Comp': 'Holl.Ra']
data_glass = pd.read_csv('glass.csv')
y_glass = data_glass['Type']
X_glass = data_glass.loc[:, 'RI': 'Fe']


def readfile(filename):
    data = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    for row in range(1, len(data)):
        for el in range(len(data[row]) - 1):
            data[row][el] = float(data[row][el])
        try:
            data[row][-1] = int(float(data[row][-1]))
        except:
            pass
    return data


Tree_Gen(X_veh, y_veh, X_glass, y_glass)
KNN_Gen(readfile('C:/Users/n.zavyalov/PycharmProjects/autodecoder/Vehicle.csv'),
        readfile('C:/Users/n.zavyalov/PycharmProjects/autodecoder/Glass.csv'))
