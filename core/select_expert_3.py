#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:57:07 2019

@author: Jiahao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:42:35 2019

@author: yitongcai,Jiahao
"""
import os
os.chdir('/scratch/mmy272/test/CapstoneProject/')

from core.experts import *
import numpy as np
import pandas as pd
import sys


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
        elif expert == "RF":
            self.model = RandomForest(*self.hyperparams) 
        elif expert == "XGBoost":
            self.model = XGBoost(*self.hyperparams) 
        else:
            raise Exception('{} has not been implemented yet'.format(self.expert))
        
        if train_data is not None:
            df_train = pd.read_csv(train_data,header=None)
            self.y_train = np.array(df_train.iloc[:,0])
            self.X_train = np.array(df_train.iloc[:,1:])
            
            del df_train
        else: 
            X_train,y_train = self.generate_data()
        
        if test_data is not None:
            df_test = pd.read_csv(test_data,header=None)
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
    model_name = sys.argv[1]
    
    model_parameters = sys.argv[2]
    
    # random round
    n_random = sys.argv[3]
    
    # number of experts
    n_experts = int(sys.argv[4])
    
    # file
    file = sys.argv[5]
    _dir = '/scratch/mmy272/test/scripts2/'
    file_obj = open(os.path.join(_dir,file),'r')
    line = file_obj.readline().split(',')
    train_data = line[0]
    test_data = line[1]
    _sigma = line[2]
    file_obj.close()
    
    

    test_expert = test_expert(model_name,model_parameters,train_data,test_data)
    losses, preds = test_expert.train_test()
    
    # write out results
    file_name = model_name + '_' + model_parameters + '.csv'
    output_directory = '/scratch/mmy272/test/output2/{}/{}/{}/'.format(_sigma,n_random,n_experts)

    output_file_path = os.path.join(output_directory,file_name)
    
    columns = ['prediction','loss']
    df_out = pd.DataFrame(np.array([preds,losses]).T,columns = columns)
    df_out.to_csv(output_file_path,index=False)