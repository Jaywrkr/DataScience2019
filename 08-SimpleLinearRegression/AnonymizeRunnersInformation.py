# import all the required libraries and put matplotlib in inline mode to plot on the notebook
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
import hashlib
import xlwt

def AnonymizeRunners(input_fn, output_fn):
    df = pd.read_excel(input_fn)

    ### utf-8 encode
    df['LastName'] = df['LastName'].apply(lambda x: x.encode('utf-8'))
    df['FirstName'] = df['FirstName'].apply(lambda x: x.encode('utf-8'))
    df['Team'] = df['Team'].apply(lambda x: x.encode('utf-8'))

    ### anonymize
    df['LastName'] = df['LastName'].apply(lambda x: str(hashlib.sha224(x).hexdigest()))
    df['FirstName'] = df['FirstName'].apply(lambda x: str(hashlib.sha224(x).hexdigest()))
    df['Team'] = df['Team'].apply(lambda x: str(hashlib.sha224(x).hexdigest()))

    writer = pd.ExcelWriter(output_fn)
    df.to_excel(writer, 'Sheet1')
    writer.save()

def AnonymizeResultFiles(input_files, output_files):

    if (len(input_files)!=len(output_files)):
        print ("ERROR: lists have different sizes")
        print ("")
        return


    for i, ifn in enumerate(input_files):

        ### output file name
        ofn = output_files[i]
        AnonymizeRunners(ifn, ofn)





