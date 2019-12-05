#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:04:40 2019

@author: Jiahao
"""

'''
This file takes hpc output, compute a regret and a percent time table
'''

import os
os.chdir('/Users/Jiahao/Documents/CapstoneProject/')

from core.online_learner_hpc import *
import pandas as pd 
import numpy as np

# different sigmas, cols
sigmas = np.array([0,1,5,10,15,20])
# number of rounds, rows
N = 2
n_rounds = np.array([i+1 for i in range(N)])

df_regret = pd.DataFrame(index = n_rounds, columns=sigmas)
df_percent = pd.DataFrame(index = n_rounds,columns=sigmas)


# online learning configure
redis = 0.4


# fill in numbers 
for _round in n_rounds:
    for sigma in sigmas:
        result_folder = '/Users/Jiahao/Desktop/output/round{}/{}/'.format(_round,sigma)
        y_test = np.array(pd.read_csv('/Users/Jiahao/Desktop/old/data//xgb_test_{}.csv'.format(sigma),header=None).iloc[:,0])
         
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
        
        df_regret.at[_round,sigma] = R
        df_percent.at[_round,sigma] = sum(W[:,best_expert])/len(W)
    # check
    print('finish {} round'.format(_round))


# write out
df_regret.to_csv('/Users/Jiahao/Desktop/regret_table.csv',index=False)
df_percent.to_csv('/Users/Jiahao/Desktop/percent_table.csv',index=False)
        
        
        


