#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:42:35 2019

@author: yitongcai,Jiahao
"""

from experts import *
from data_generator import *
import numpy as np
import pandas as pd
import sys
import os

'''
This file is to be used on hpc

We use it to test single expert.
This file reports single expert's prediction and loss on testing data
'''
class test_expert(object):
    
    def __init__(self, expert,hyperparams,train_data,test_data):
        
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
            
        self.hyperparams = [float(p) if is_number(p) else p for p in hyperparams.split(",") ]  #[hyper1,hyper2,hyper3,...]
        
        if expert == "AR":
            self.model = AR(self.hyperparams[0])
        elif expert == "LR":
            self.model = LinearRegression()
        elif expert == "SVR":
            self.model = SVR(*self.hyperparams) 
        else:
            raise Exception('{} has not been implemented yet'.format(self.expert))
        
        if train_data is not None:
            df_train = pd.read_csv(train_data)
            self.y_train = np.array(df_train.iloc[:,0])
            self.X_train = np.array(df_train.iloc[:,1:])
            
            del df_train
        else: 
            X_train,y_train = self.generate_data()
        
        if test_data is not None:
            df_test = pd.read_csv(test_data)
            self.y_test = np.array(df_test.iloc[:,0])
            self.X_test = np.array(df_test.iloc[:,1:])
            
            del df_test
        else:
            X_test,y_test = self.generate_data()
        
    def generate_data():
        '''
        If no data path is provided, this module generate its own data
        '''
        N = 1000
        sigma = 1
        kernel = 'linear_regression' 
        parameter = [1,[1,2,1.5]]
        generator = data_generator(kernel,parameter,sigma,N)
        return generator.generate()
    
    def train_test(self):
        # train
        self.model.train(self.X_train,self.y_train)
        
        # test
        preds, losses=[], []
        for x,y in zip(self.X_test,self.y_test):
            y_pred = self.model.predict(x)
            loss = (y - y_pred)**2
            losses.append(loss)
            preds.append(y_pred)
        return (losses, preds)
    
if __name__ == '__main__':
    '''
    Input argvs:  expert name, hyperparameter, train data path,
                  test data path, output directory path
    
    '''
    # test expert
    model_parameters = sys.argv[1:-1]
    test_expert = test_expert(*model_parameters)
    losses, preds = test_expert.train_test()
    
    # write out results
    output_directory = sys.argv[-1]
    file_name = model_parameters[0] + '_' + model_parameters[1] + '.csv'
    output_file_path = os.path.join(output_directory,file_name)
    
    columns = ['prediction','loss']
    df_out = pd.DataFrame(np.array([preds,losses]).T,columns = columns)
    df_out.to_csv(output_file_path,index=False)
    
    





   
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
             
             
             
             
             
             
             
             
         