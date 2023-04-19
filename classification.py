# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 00:41:23 2022

@author: mighty
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#obj to load book
a = pd.read_csv(r'E:\Folders\thapar waala maal\sem 5\Machine Learning\project files\Book1.csv', lineterminator='\n')
#a is the dataframe
scale = MinMaxScaler()

read = np.array(float(input('cost:\n')))
read1 = read.reshape(-1,1)

ar = a.columns.get_loc("cost")
ar2 = a.columns.get_loc("costb")
X = a.iloc[:,ar:ar2]

Y = a.iloc[:,a.columns.get_loc("brand_id"):a.columns.get_loc("brand_id")+1]

a['scaled']=Y

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.3,train_size = 0.7, random_state = True, shuffle = True)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train,Y_train)
Y_pred = model.predict(read1)

print(Y_pred)
