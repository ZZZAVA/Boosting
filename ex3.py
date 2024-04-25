import pandas as pd

from AdaBoost import AdaBoost
from KNN import KNN_Gen

data = pd.read_csv('Vehicle.csv')
y_veh = data['Class']
X_veh = data.loc[:, 'Comp': 'Holl.Ra']
data_glass = pd.read_csv('glass.csv')
y_glass = data_glass['Type']
X_glass = data_glass.loc[:, 'RI': 'Fe']

# AdaBoost
AdaBoost(X_veh, y_veh, X_glass, y_glass)

# KNN
KNN_Gen(X_veh, y_veh, X_glass, y_glass)