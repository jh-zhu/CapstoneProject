#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:42:35 2019

@author: yitongcai
"""
from core.experts import *
import numpy as np

'''
python expert.py(0) SVR(1) 1,2,3(2)
import sys
model, hyper = sys.argv[1], sys.argv[2]
test_expert (import expert.py, take data), arg {model, hyper}  ->  loss 
'''
class test_expert(object):
    
    def  __init__(self, expert, hyperparams):
        self.expert = expert
        self.hyperparams = [int(p) if p.isdigit() else p for p in hyperparams.split(",") ]  #[hyper1,hyper2,hyper3,...]
        
        if expert == "AR":
            self.model = AR(self.hyperparams[0])
        elif expert == "LR":
            self.model = LinearRegression()
        elif expert == "SVR":
            self.model = SVR(*self.hyperparams) 
        else:
            raise Exception('{} has not been provided yet'.format(self.expert))
            
    def train_test(self):
        '''Get the real dataset with X, y and split into train set and test set'''
        
        
        '''Generate data from specific model for training'''
        sigmas=np.arange(0,10,1)
        N = 1000 # number of data to generate
        noise_level = 1 # level of noise (sigma), added to the generating process
        kernel = 'linear_regression' # the model to generate 
        parameter = [1,[1,2,1.5],noise_level,N] # parameter of data generating model
        
        generator = data_generator(kernel,parameter,noise_level,N)
        X_train,y_train = generator.generate()
        X_test,y_test = generator.generate()
        self.model.train(X_train,y_train)
        
        losses=[]
        for x,y in zip(X_test,y_test):
            y_pred = self.model.predict(x)
            MSE = abs(y - y_pred)**2
            losses.append(MSE)
        return losses
            
import sys
expert, hyper = sys.argv[1], sys.argv[2]   
test_expert = test_expert(expert, hyper)
test_expert.train_test()

        
    
#    def RandomCreate(self, **kwargs):
#        ''' Use the random grid to search for best hyperparameters'''
#        random_range = {k:v for k, v in kwargs.items()}
#        
#        ARs = [AR(p) for p in ps]
#        SARIMAXs = [SARIMAX(p,d,q,P,D,Q,m) for (p,d,q,P,D,Q,m) in zip(*((ps,ds,qs,Ps,Ds,Qs,ms)))]
#        LRs = [LinearRegression()]
#        SVRs = [SVR(kernel, gamma, C, epsilon) for (kernel, gamma, C, epsilon) in zip(*(kernels, gammas, Cs, epsilons))]
#        
#        RFs=[]
#        Boosts=[]
#        LSTMs=[]
#        
#        self.models =  ARs + SARIMAXs + LRs + SVRs 
#        self.model_names=[m.name for m in self.models]
        


#'''AR'''  
#ps = 
#     
#'''SARIMAX''' 
#ps = np.arange(0,)
#ds = 
#qs = 
#Ps = 
#Ds = 
#Qs = 
#ms =
#
#'''SVR'''
#kernels =
#gammas = 
#Cs = 
#epsilons =       
#        
# '''Random Forest''' 
# # Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
## Number of features to consider at every split
#max_features = ['auto', 'sqrt']
## Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
## Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10]
## Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
## Method of selecting samples for training each tree
#bootstrap = [True, False]
##First create the base model to tune
#rf = RandomForestRegressor(random_state = 42) 
#
#  
#hyper_opt.RandomSearch(rf, n_estimators=n_estimators, max_features=max_features, max_depth = max_depth,
#             min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,bootstrap=bootstrap)
#             
#             
#             
             
             
             
             
             
             
             
             
         