#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:14:56 2019

@author: Jiahao


Generate data set using SVR 
"""
import os
os.chdir('/Users/Jiahao/Documents/classes/capstone/online_learning/')

from core.experts import *
import pandas as pd 
import numpy as np



sigma = 50

xgb_hyperparameter = [3, 0.05, 100, 1, 1, 
                 0, 0, 0]
xgb_model = XGBoost(*xgb_hyperparameter)

train_file = '/Users/Jiahao/Downloads/train.csv'
test_file = '/Users/Jiahao/Downloads/test.csv'
df_train = pd.read_csv(train_file,header=None)
df_test = pd.read_csv(test_file,header=None)

l = len(df_train)

X_train = np.array(df_train.iloc[:,1:])
X_test = np.array(df_test.iloc[:,1:])
y_train = np.array(df_train.iloc[:,0])
y_test = np.array(df_test.iloc[:,0])

# first fit SVR model 
xgb_model.train(X_train,y_train)
# do insample predict
y_train_hat = xgb_model.predict_batch(X_train) + np.random.normal(0,sigma,l)


xgb_model.train(X_test,y_test)
y_test_hat = xgb_model.predict_batch(X_test) + np.random.normal(0,sigma,l)

df_train_out = df_train.copy()
df_test_out = df_test.copy()

df_train_out.iloc[:,0] = y_train_hat
df_test_out.iloc[:,0] = y_test_hat

df_train_out.to_csv('/Users/Jiahao/Downloads/xgb_train_{}.csv'.format(sigma),index=False,header=False)
df_test_out.to_csv('/Users/Jiahao/Downloads/xgb_test_{}.csv'.format(sigma),index=False,header=False)










 






