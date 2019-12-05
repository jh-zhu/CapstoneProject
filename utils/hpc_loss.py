#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:04:40 2019

@author: Jiahao
"""

'''
This file takes hpc output, compute a loss table
'''

import os
os.chdir('/Users/Jiahao/Documents/CapstoneProject/')

from core.online_learner_hpc import *
import pandas as pd 
import numpy as np

# different sigmas, cols
n_experts = np.array([1000,500,200,100])
# number of rounds, rows
N = 2
n_rounds = np.array([i+1 for i in range(N)])


# mode
mode = 'grid'
sigma = 5

df_loss = pd.DataFrame(index = n_rounds, columns=n_experts)


# online learning configure
redis = 0.4


# fill in numbers 
for _round in n_rounds:
    for n in n_experts:
        result_folder = '/Users/Jiahao/Desktop/output/round{}/{}/{}/{}'.format(_round,sigma,mode,n)
        y_test = np.array(pd.read_csv('/Users/Jiahao/Desktop/old/data/xgb_test_{}.csv'.format(sigma),header=None).iloc[:,0])
         
        # create an online learner
        learner = FTL_hpc(source_path = result_folder,redis = redis)
        
        # get expert weights change matrix
        W = learner.compute_weight_change()
        # get algo prediction over time 
        P = learner.compute_algo_prediction(W)
        # get algo loss over time 
        L = learner.compute_algo_loss(P,y_test)
        # get algo regret
        R,best_expert = learner.compute_algo_regret(L)
        
        df_loss.at[_round,n] = sum(L)
    # check
    print('finish {} round'.format(_round))


# write out
df_loss.to_csv('/Users/Jiahao/Desktop/regret_loss.csv',index=False)