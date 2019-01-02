import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model


def compute_polynomial_model(x, coef, intercept):
    min_x = min(x)
    max_x = max(x)
    xp = np.arange(min_x, max_x, (max_x-min_x)/100.0)

    x = xp
    yp = intercept

    for w in coef:
        yp = yp + w * x
        x = x * xp
    return xp,yp


def add_degrees(dataset,original_column, degree):
    new_dataset = dataset
    current_column = original_column
    for d in range(2,degree+1):
        column_name = original_column+str(d)

        new_dataset[column_name] = new_dataset[current_column]*dataset[original_column]

        current_column = column_name

    return new_dataset

def generate_variables(original_column, degree):
    v = [original_column]
    for d in range(2,degree+1):
        v.append(original_column+str(d))
    return v

def plot_approximator(x,y,xp,yp, title):
    font = {'family' : 'sans',
        'size'   : 16}
    plt.rc('font', **font)

    plt.scatter(x, y,  color='blue')
    plt.plot(xp, yp, color='red', linewidth=3)
    plt.xlabel("LSTAT")
    plt.ylabel("MEDV")

    plt.xlim([0,40])
    plt.ylim([0,60])
    plt.show()

def rss(y,yp):
    d = y-yp;
    return d*d

def pretty_print(x):
    if (x>0):
        return " + " + str(round(x,3))
    else:
        return str(round(x,3))

def compute_polynomial_regression(dataset, variable, target, degree):

    extended_dataset = add_degrees(dataset, variable, degree)

    dataset_train_x = extended_dataset[generate_variables(variable,degree)].values
    dataset_train_y = extended_dataset[target].values

    x = dataset_train_x.reshape(506, degree)
    y = dataset_train_y.reshape(506, 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    yp = regr.predict(x)

    rss = sum((yp-y)*(yp-y))
    r2 = regr.score(x,y)

    # the abscissa
    xm,ym = compute_polynomial_model(x[:,0],regr.coef_[0], regr.intercept_)

    return xm, ym, yp, rss[0],r2


dataset = pd.read_csv('housing.csv')

### ============================================================
### SIMPLE LINEAR REGRESSION
### ============================================================

# LSTAT - % lower status of the population
dataset_train_x = dataset[generate_variables('LSTAT',1)].values

# MEDV - Median value of owner-occupied homes in $1000's
dataset_train_y = dataset.MEDV.values

x = dataset_train_x.reshape(506, 1)
y = dataset_train_y.reshape(506, 1)

regr = linear_model.LinearRegression()
regr.fit(x, y)

# the abscissa
xp,yp = compute_polynomial_model(x[:,0],regr.coef_, regr.intercept_)

plot_approximator(x[:,0],y,xp,yp)

### ============================================================
### 2ND DEGREE POLYNOMIAL
### ============================================================

dataset_2nd = add_degrees(dataset, 'LSTAT', 2)

# LSTAT - % lower status of the population
dataset_2nd_train_x = dataset[generate_variables('LSTAT',2)].values

# MEDV - Median value of owner-occupied homes in $1000's
dataset_2nd_train_y = dataset.MEDV.values

x = dataset_2nd_train_x.reshape(506, 2)
y = dataset_2nd_train_y.reshape(506, 1)

regr = linear_model.LinearRegression()
regr.fit(x, y)

# the abscissa
xp2,yp2 = compute_polynomial_model(x[:,0],regr.coef_[0], regr.intercept_)

plot_approximator(x[:,0],y,xp2,yp2)


### ============================================================
### 3ND DEGREE POLYNOMIAL
### ============================================================

dataset_3rd = add_degrees(dataset, 'LSTAT', 3)

# LSTAT - % lower status of the population
dataset_train_x = dataset[generate_variables('LSTAT',3)].values

# MEDV - Median value of owner-occupied homes in $1000's
dataset_train_y = dataset.MEDV.values

x = dataset_train_x.reshape(506, 3)
y = dataset_train_y.reshape(506, 1)

regr = linear_model.LinearRegression()
regr.fit(x, y)

# the abscissa
xp2,yp2 = compute_polynomial_model(x[:,0],regr.coef_[0], regr.intercept_)

plot_approximator(x[:,0],y,xp2,yp2)

def pretty_print(intercept, coeff, variables, target):

    eq = target + " = " + str(intercept)

    for v in range(len(coeff)):
        sym = " "
        if coeff[v]>0:
            sym = " + "
        eq = eq + sym + str(round(coeff[v],2)) + variables[v]

        if (len(eq)>60):
            print(eq)
            eq = " " * (len(target + " = "))
            #eq = eq + "\n" + " "*(len(target + " = ")

    print(eq)

### ============================================================
### MULTIPLE REGRESSION
### ============================================================

variables = dataset.columns.tolist()
variables.remove('MEDV')

dataset_train_x = dataset[variables].values
dataset_train_y = dataset['MEDV'].values

x = dataset_train_x.reshape(len(dataset), len(variables))
y = dataset_train_y.reshape(len(dataset), 1)

multiregr = linear_model.LinearRegression()
multiregr.fit(x, y)

eq = pretty_print(multiregr.intercept_[0], multiregr.coef_[0], variables, 'MEDV')

### CLUSTERING

dataset = pd.read_csv('housing.csv')
dataset.describe()
plt.scatter(dataset['INDUS'], dataset['ZN'],  color='blue')


def PolynomialRegression(dataset, input, output, degree):

    if (degree==1):
        dataset_train_x = dataset[input].values
        x = dataset_train_x.reshape(len(dataset), 1)

        dataset_train_y = dataset[output].values
        y = dataset_train_y.reshape(len(dataset), 1)
    else:
        new_dataset = add_degrees(dataset, input, degree)

        dataset_train_x = new_dataset[generate_variables(input, degree)].values
        x = dataset_train_x.reshape(len(new_dataset), degree)

        dataset_train_y = dataset[output].values
        y = dataset_train_y.reshape(len(new_dataset), 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    return regr,x,y,

# for i in range(5):
regr1 = PolynomialRegression(dataset,'LSTAT','MEDV',1)
regr2 = PolynomialRegression(dataset,'LSTAT','MEDV',2)


from sklearn.model_selection import cross_val_score

def compute_polynomial_regression_cv(dataset, variable, target, degree):

    extended_dataset = add_degrees(dataset, variable, degree)

    dataset_train_x = extended_dataset[generate_variables(variable,degree)].values
    dataset_train_y = extended_dataset[target].values

    x = dataset_train_x.reshape(len(extended_dataset), degree)
    y = dataset_train_y.reshape(len(extended_dataset), 1)

    tss = sum((y - (sum(y)/float(len(y))))**2)[0]


    regr = linear_model.LinearRegression()
    scores = cross_val_score(regr, x, y, cv=10)


    rss = (1 - scores)*tss

    return (sum(scores)/10.0), (sum(rss)/10.0)

compute_polynomial_regression_cv(dataset,'LSTAT','MEDV',2)

max_polynomial = 15
rss = np.zeros(max_polynomial)
r2 = np.zeros(max_polynomial)

for i in range(max_polynomial):
    r2[i],rss[i] = compute_polynomial_regression_cv(dataset,'LSTAT','MEDV',i+1)



### CLASSIFICATION

min_x = min(X[:,0])
max_x = max(X[:,0])
xp = np.arange(min_x, max_x, (max_x-min_x)/100.0)
Y = logistic.intercept_[0]/-logistic.coef_[0][1] + xp*logistic.coef_[0][0]/-logistic.coef_[0][1]



###
###
###

def compute_polynomial_regression_holdout(dataset, variable, target, degree, test_size=0.33, random_state=1234):
    '''Computes RSS and R2 statistics over the train and test set.

    test_size is the percentage of data used for testing (default is 1/3)
    random_state is the seed used for sampling the data and it is used for replicability

    '''
    extended_dataset = add_degrees(dataset, variable, degree)

    ### Split train and test
    train_data, test_data = sklearn.model_selection.train_test_split(extended_dataset, test_size=0.33, random_state=1234)

    train_x = train_data[generate_variables(variable,degree)].values
    train_y = train_data[target].values

    test_x = test_data[generate_variables(variable,degree)].values
    test_y = test_data[target].values

    # training data
    train_x = train_x.reshape(len(train_x), degree)
    train_y = train_y.reshape(len(train_y), 1)

    # test data
    test_x = test_x.reshape(len(test_y), degree)
    test_y = test_y.reshape(len(test_y), 1)

    tss_train = sum((train_y - (sum(train_y)/float(len(train_y))))**2)[0]
    tss_test = sum((test_y - (sum(test_y)/float(len(test_y))))**2)[0]

    regr = linear_model.LinearRegression()

    regr.fit(train_x, train_y)

    y_predicted_from_train = regr.predict(train_x)
    rss_train = sum( (train_y-y_predicted_from_train)*(train_y-y_predicted_from_train) )[0]

    y_predicted_from_test = regr.predict(test_x)
    rss_test = sum( (test_y-y_predicted_from_test)*(test_y-y_predicted_from_test) )[0]

    return rss_train, rss_test, (1-rss_train/tss_train), (1-rss_test/tss_test)


compute_polynomial_regression_holdout(dataset,'LSTAT','MEDV',2)

max_polynomial = 15
rss_train = np.zeros(max_polynomial)
r2_train = np.zeros(max_polynomial)
rss_test = np.zeros(max_polynomial)
r2_test = np.zeros(max_polynomial)

for i in range(max_polynomial):
    rss_train[i],rss_test[i],r2_train[i],r2_test[i] = compute_polynomial_regression_holdout(dataset,'LSTAT','MEDV',i+1)

plt.plot(range(max_polynomial+1),rss_train)
