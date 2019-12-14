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

sigma = 1
source_path = '/Users/yitongcai/Coding/Capstone_data/output2/{}/G/'.format(sigma)
y_test = np.array(pd.read_csv("/Users/yitongcai/Coding/Capstone_data/round_1/xgb_test_{}.csv".format(sigma), header=None).iloc[:,0])

'''GRID SEARCH'''
algo_abs_loss_grid=[]
num_of_experts=[100, 200, 400, 600, 800]

for f in num_of_experts:
#        for ff in sorted(os.listdir(source_path+f+'/')):
    # create a online learner calculator
    redis = 0.7
#        learning_rate = 0.5
    OL_name= "Follow the Leader"
    learner = FTL_hpc(source_path = source_path+str(f)+'/',
#                          learning_rate = learning_rate,
                      redis = redis)
    
    
    # get expert weights change matrix
    W = learner.compute_weight_change()
#            W = learner.compute_underlying_weight_change()
    
    # find the leading expert over time
    lead_expert = learner.find_leading_expert(W)
    
    # get algo prediction over time 
    P = learner.compute_algo_prediction(W)
    # get algo loss over time 
    L = learner.compute_algo_loss(P,y_test)
    
    algo_abs_loss_grid.append(sum(L))           

algo_abs_loss_grid=[9447.5, 8402.81, 7855.63, 7352.33, 7349.54]            
fig = plot_random_grid(num_of_experts, algo_abs_loss_grid,
                       title = OL_name+" - Grid Search"
#                       +" with threshold"
                       )

'''RANDOM SEARCH''' 
algo_abs_loss_random=[0, 0, 0, 0, 0]
sigma = 1
source_path = '/Users/yitongcai/Coding/Capstone_data/output3/{}/'.format(sigma)
y_test = np.array(pd.read_csv("/Users/yitongcai/Coding/Capstone_data/round_1/xgb_test_{}.csv".format(sigma), header=None).iloc[:,0])

for i in range(20):
    for j in range(len(num_of_experts)):
        # create a online learner calculator
        redis = 0.4
#        learning_rate = 0.5
        OL_name= "Follow the Leader"
        learner = FTL_hpc(source_path = source_path+str(i)+'/'+str(num_of_experts[j])+'/',
#                          learning_rate = learning_rate,
                          redis = redis)
        
        
        # get expert weights change matrix
        W = learner.compute_weight_change()
#            W = learner.compute_underlying_weight_change()
        
        # find the leading expert over time
        lead_expert = learner.find_leading_expert(W)
        
        # get algo prediction over time 
        P = learner.compute_algo_prediction(W)
        # get algo loss over time 
        L = learner.compute_algo_loss(P,y_test)
        
        algo_abs_loss_random[j]+=sum(L)      
        
algo_abs_loss_random = [i/20 for i in algo_abs_loss_random]

algo_abs_loss_random=[21914.465176982194, 8644.052460114452,
 8243.388048328592,
 7807.099754717834,
 7786.683952790476]            
fig = plot_random_grid(num_of_experts, algo_abs_loss_random,
                       title = OL_name+" - Random Search"
#                       +" with threshold"
                       )
          
