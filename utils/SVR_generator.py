#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:14:56 2019

@author: Jiahao


Generate data set using SVR 
"""

from core.experts import *
import pandas as pd 
import numpy as np

sigma = 10 

hypyerparameter_A = ['linear',0.01,0.1]
hyperparameter_B = ['rbf',0.03,0.4]

SVR_A = SVR(*hypyerparameter_A)
SVR_B = SVR(*hyperparameter_B)


data_file = '/Users/Jiahao/Downloads/train.csv'
df_train = pd.read_csv(data_file,header=None)
l = len(df_train)
X = np.array(df_train.iloc[:,1:])
y = np.array(df_train.iloc[:,0])

# first fit SVR model 
SVR_A.train(X,y)
SVR_B.train(X,y)

# do insample predict
y_A = SVR_A.predict_batch(X) + np.random.normal(0,sigma,l)
y_B = SVR_B.predict_batch(X) + np.random.normal(0,sigma,l)

df_A = df_train.copy()
df_B = df_train.copy()
df_A.iloc[:,0] = y_A
df_B.iloc[:,0] = y_B

df_output = pd.concat([df_A,df_B])

df_output.to_csv('/Users/Jiahao/Documents/SVR_{}.csv'.format(sigma),index=False,header=False)










 






