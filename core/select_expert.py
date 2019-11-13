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
            self.model = LinearRegression(*self.hyperparams)
        elif expert == "SVR":
            self.model = SVR(*self.hyperparams) 
        else:
            raise Exception('{} has not been implemented yet'.format(self.expert))
        
        if train_data is not None:
            df_train = pd.read_csv(train_data,header=False)
            self.y_train = np.array(df_train.iloc[:,0])
            self.X_train = np.array(df_train.iloc[:,1:])
            
            del df_train
        else: 
            X_train,y_train = self.generate_data()
        
        if test_data is not None:
            df_test = pd.read_csv(test_data,header=False)
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
             
             
             
             
             
             
             
         