#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:26:04 2019

@author: Jiahao
"""


import pandas as pd 
import numpy as np

# number of rounds
N = 100
rounds = np.array([i+1 for i in range(N)])
sigmas = np.array([0,1,5,10,15,20])

xgb_hyperparameter = [3, 0.05, 100, 1, 1, 
                     0, 0, 0]
xgb_model_train = XGBoost(*xgb_hyperparameter)
xgb_model_test = XGBoost(*xgb_hyperparameter)



train_file = '/Scratch/mmy272/test/CapstoneProject/init_data/train.csv'
test_file = '/Scratch/mmy272/test/CapstoneProject/init_data/test.csv'
df_train = pd.read_csv(train_file,header=None)
df_test = pd.read_csv(test_file,header=None)

l = len(df_train)

X_train = np.array(df_train.iloc[:,1:])
X_test = np.array(df_test.iloc[:,1:])
y_train = np.array(df_train.iloc[:,0])
y_test = np.array(df_test.iloc[:,0])

xgb_model_train.train(X_train,y_train)
xgb_model_test.train(X_test,y_test)

for _round in rounds:
    for sigma in sigmas:
        y_train_hat = xgb_model_train.predict_batch(X_train) + np.random.normal(0,sigma,l)
        y_test_hat = xgb_model_test.predict_batch(X_test) + np.random.normal(0,sigma,l)
        
        df_train_out = df_train.copy()
        df_test_out = df_test.copy()
        
        df_train_out.iloc[:,0] = y_train_hat
        df_test_out.iloc[:,0] = y_test_hat
        
        df_train_out.to_csv('/scratch/mmy272/test/data/round_{}/xgb_train_{}.csv'.format(_round,sigma),index=False,header=False)
        df_test_out.to_csv('/Users/Jiahao/Downloads/round_{}/xgb_test_{}.csv'.format(_round,sigma),index=False,header=False)
