#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:53:44 2019

@author: yitongcai
"""
import numpy as np
import pandas as pd
import os
from core.online_learner_hpc import *
from utils.plot import *

sigma = 0
source_path = '/Users/yitongcai/Coding/output/{}/'.format(sigma)
y_test = np.array(pd.read_csv("/Users/yitongcai/Coding/data/xgb_test_{}.csv".format(sigma), header=None).iloc[:,0])

'''GRID SEARCH'''
#algo_abs_loss_grid=[]
#num_of_experts=[]
#
#for f in sorted(os.listdir(source_path)):
#    if not f.startswith('.'):
#        for ff in sorted(os.listdir(source_path+f+'/')):
#            # create a online learner calculator
#            redis = 1
#            learning_rate = 0.5
#            OL_name= "RWM"
#            learner = RWM_hpc(source_path = ff,
#                              learning_rate = learning_rate,
#                              redis = redis)
#            
#            
#            # get expert weights change matrix
#            W = learner.compute_weight_change()
##            W = learner.compute_underlying_weight_change()
#            
#            # find the leading expert over time
#            lead_expert = learner.find_leading_expert(W)
#            
#            # get algo prediction over time 
#            P = learner.compute_algo_prediction(W)
#            # get algo loss over time 
#            L = learner.compute_algo_loss(P,y_test)
#            
#            algo_abs_loss_grid.append(L[-1])
#            num_of_experts.append(len(os.listdir(source_path+f+'/')))
#            
#
#            
#fig = plot_random_grid(num_of_experts, algo_abs_loss_grid,
#                       title = OL_name+"  grid_search  "+str(redis)+"  "+str(sigma)
##                       +" with threshold"
#                       )
num_of_experts=[10, 50, 100, 500, 1000]

'''RANDOM SEARCH''' 
algo_abs_loss_random=[]

for num in num_of_experts:
    # create a online learner calculator
    redis = 1
    learning_rate = 0.5
    OL_name= "RWM"
    learner = RWM_hpc(source_path = source_path,
                      learning_rate = learning_rate,
                      redis = redis, num_experts=num)
    
    
    # get expert weights change matrix
    W = learner.compute_weight_change()
#            W = learner.compute_underlying_weight_change()
    
    # find the leading expert over time
    lead_expert = learner.find_leading_expert(W)
    
    # get algo prediction over time 
    P = learner.compute_algo_prediction(W)
    # get algo loss over time 
    L = learner.compute_algo_loss(P,y_test)
    
    algo_abs_loss_random.append(L[-1])
        
            
fig = plot_random_grid(num_of_experts, algo_abs_loss_random,
                       title = OL_name+"  random_search  "+str(redis)+"  "+str(sigma)
#                       +" with threshold"
                       )
            
import random
import itertools          
max_depth2=[int(x) for x in np.linspace(start = 3, stop = 7, num = 4)]
learning_rate=np.linspace(0.01,0.2,4)
n_estimators2=[100,250,500,1000]
subsample=np.linspace(0.5,1,4)
colsample_bytree=np.linspace(0.5,1,4)
gamma2=[0.01,1]
alpha2=[0.0,1]
lambd=[0.0,5] 

samples = random.sample(set(itertools.product(max_depth2, learning_rate, n_estimators2,
                                          subsample, colsample_bytree, gamma2, 
                                          alpha2, lambd)), 10)
print(samples)
          
