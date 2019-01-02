# import all the required libraries and put matplotlib in inline mode to plot on the notebook
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
import hashlib
import xlwt

import AnonymizeRunnersInformation

def Genre(category):
    '''Computes the genre from the FIDAL running category
    http://www.fidal.it/content/Le-categorie-di-tesseramento-atleti/49913
    '''
    if category==None:
        return
    if category=='PM' or category=='SM':
        return 'Male'
    elif category=='PF' or category=='SF':
        return 'Female'
    elif str(category)[0:2]=='SM':
        return 'Male'
    else:
        return 'Female'

def Age(category):
    '''Computes the age from the FIDAL running category
    http://www.fidal.it/content/Le-categorie-di-tesseramento-atleti/49913
    '''

    if category==None:
        return 0.0

    # promesse
    if category=='PM' or category=='PF':
        return 20.0

    # senior < 35
    if category=='SM' or category=='SF':
        return 23.0

    s = str(category)

    return float(s[2:4])

def MaxAge(category):
    '''Computes the upper bound to runner's age from the FIDAL running category
    http://www.fidal.it/content/Le-categorie-di-tesseramento-atleti/49913
    '''

    if category==None:
        return 0.0

    # promesse
    if category=='PM' or category=='PF':
        return 22.0

    # senior < 35
    if category=='SM' or category=='SF':
        return 35.0

    s = str(category)

    return float(s[2:4])+4

def ConvertTimeToSeconds(time_string):
    '''Computes the running time from hh:mm:ss format'''

    st = str(time_string).split(':')

    # less than a minute
    if (len(st)==1):
        st = ['0'] + st

    # less than a hour
    if (len(st)==2):
        st = ['0'] + st

    return float(st[0])*3600 + float(st[1])*60 +float(st[2])

def ConvertDelay(dt):
    '''Computes the delay from the first runner time from +hh:mm:ss format'''
    return ConvertTimeToSeconds(str(dt)[1:])


def PreprocessRunnersStats(df, distance=42.198):

    cdf = df

    cdf['Age'] = cdf['Category'].apply(Age)
    cdf['MaxAge'] = cdf['Category'].apply(MaxAge)
    cdf['Genre'] = cdf['Category'].apply(Genre)
    cdf['TimeInSeconds'] = cdf['OfficialTime'].apply(ConvertTimeToSeconds)
    cdf['TimeInHours'] = cdf['TimeInSeconds']/3600.0
    cdf['MinutesPerKm'] = (cdf['TimeInSeconds']/60.0)/distance
    cdf['DelayInSeconds'] = cdf['Delay'].apply(ConvertDelay)

    return cdf

def ComputeStats(df, target):

    age = []
    q1 = []
    q2 = []
    q3 = []
    avg = []
    stddev = []
    stderr = []

    unique_age = sorted(df['Age'].unique())

    for a in unique_age:
        # selected data
        sd = np.array(df[df['Age']==a][target])

        age.append(a)
        q1.append(np.percentile(sd,25))
        q2.append(np.percentile(sd, 50))
        q3.append(np.percentile(sd, 75))
        avg.append(np.average(sd))
        stddev.append(np.std(sd))
        stderr.append(np.std(sd)/len(sd))

        #print(a,"\t",np.percentile(sd,25),"\t",np.percentile(sd,50),"\t",np.percentile(sd,75))

    return pd.DataFrame({'Age':age, 'FirstQuartile':q1, 'SecondQuartile':q2, 'ThirdQuartile':q3, 'Average':avg, 'StdDev':stddev, 'StdErr':stderr})


def JitterAge(age):
    if (age>=35):
        return age + random.uniform(0,5)
    if (age==20):
        return age + random.uniform(0,2)
    if (age==23):
        return age + random.uniform(0,12)


