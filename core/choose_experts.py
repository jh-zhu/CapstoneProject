#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:42:35 2019

@author: yitongcai
"""
from core.experts import *

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV


class experts_create(object):
    
    def  __init__(self, base_model, X_train, y_train, **kwargs):
        self.base_model = base_model
        self.X_train = X_train
        self.y_train = y_train
        self.models = None
        self.model_names = None
        
        
    def RandomCreate(self, **kwargs):
        ''' Use the random grid to search for best hyperparameters'''
        random_range = {k:v for k, v in kwargs.items()}
        
        ARs = [AR(p) for p in ps]
        SARIMAXs = [SARIMAX(p,d,q,P,D,Q,m) for (p,d,q,P,D,Q,m) in zip(*((ps,ds,qs,Ps,Ds,Qs,ms)))]
        LRs = [LinearRegression()]
        SVRs = [SVR(kernel, gamma, C, epsilon) for (kernel, gamma, C, epsilon) in zip(*(kernels, gammas, Cs, epsilons))]
        
        RFs=[]
        Boosts=[]
        LSTMs=[]
        
        
        self.models =  ARs + SARIMAXs + LRs + SVRs 
        self.model_names=[m.name for m in self.models]
        
    def GridCreate(self, **kwargs):
        grid_range = {k:v for k, v in kwargs.items()}

'''
'''AR'''  
ps = 
     
'''SARIMAX''' 
ps = np.arange(0,)
ds = 
qs = 
Ps = 
Ds = 
Qs = 
ms =

'''SVR'''
kernels =
gammas = 
Cs = 
epsilons =       
        
 '''Random Forest''' 
 # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#First create the base model to tune
rf = RandomForestRegressor(random_state = 42) 

 
hyper_opt = Hyperparams_Optimization(rf, X_train, y_train)    
hyper_opt.RandomSearch(rf, n_estimators=n_estimators, max_features=max_features, max_depth = max_depth,
             min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,bootstrap=bootstrap)
'''             
             
             
             
             
             
             
             
             
             
             
         