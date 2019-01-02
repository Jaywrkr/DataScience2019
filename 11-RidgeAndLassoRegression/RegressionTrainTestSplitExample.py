from sklearn import model_selection, linear_model
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

dataset = pd.read_csv('housing.csv')
dataset.columns
dataset.describe()

train_data, test_data = sklearn.model_selection.train_test_split(dataset, test_size=0.33, random_state=1234)



