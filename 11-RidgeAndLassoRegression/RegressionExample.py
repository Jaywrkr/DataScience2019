import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model

dataset = pd.read_csv('housing.csv')

# LSTAT - % lower status of the population
dataset_train_x = dataset.LSTAT.values

# MEDV - Median value of owner-occupied homes in $1000's
dataset_train_y = dataset.MEDV.values

x = dataset_train_x.reshape(506, 1)
y = dataset_train_y.reshape(506, 1)

regr = linear_model.LinearRegression()
regr.fit(x, y)


font = {'family' : 'sans',
        'size'   : 16}
plt.rc('font', **font)
plt.xlabel("LSTAT")
plt.ylabel("MEDV")

plt.scatter(x, y,  color='blue')
plt.xlim([0,40])
plt.ylim([0,60])
plt.show()


plt.plot(x, regr.predict(x), color='red', linewidth=3)

coeff = round(regr.coef_[0][0],3)
intercept = round(regr.intercept_[0],3)

eq = "MEDV = " + str(intercept)
if (coeff>0):
    eq = eq + " + "
eq = eq + str(coeff)+"*LSTAT"

plt.title (eq)
plt.xlim([0,40])
plt.ylim([0,60])
plt.show()



