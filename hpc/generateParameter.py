#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:26:11 2019

@author: mingmingyu
"""

import numpy as np


def gen_params(nums, expert):
    if expert == "LR" and len(nums)==2:
        alphas=generateParameter(-4,2,nums[0]).grid_log()
        l1s=generateParameter(0,1,nums[1]).grid_lin()
        return alphas,l1s
    
    elif expert == "SVR" and len(nums)==3:
        gammas = generateParameter(-4,3,nums[0]).grid_log()
        Cs = generateParameter(-3,4,nums[1]).grid_log()
        epsilons = generateParameter(-6,2,nums[2]).grid_log(base=2)
        return gammas, Cs, epsilons

    elif expert == "RF" and len(nums)==4:
        n_estimators = generateParameter(100,1000,nums[0]).grid_lin("int")
        max_depth = generateParameter(2,15,nums[1]).grid_lin("int")
        min_samples_split = generateParameter(2,15,nums[2]).grid_lin("int")
        min_samples_leaf = generateParameter(1,10,nums[3]).grid_lin("int")
        return n_estimators, max_depth, min_samples_split, min_samples_leaf

    elif expert == "XGBoost" and len(nums)==8:
        max_depth=generateParameter(2,4,nums[0]).grid_lin("int")
        learning_rate=generateParameter(-2,-1,nums[1]).grid_log()
        n_estimators = generateParameter(100,800,nums[2]).grid_lin("int")
        subsample = generateParameter(0.75,1,nums[3]).grid_lin()
        colsample_bytree = generateParameter(0.75,1,nums[4]).grid_lin()
        gamma = generateParameter(-2,0,nums[5]).grid_log()
#        alpha =  generateParameter(0.2,0.6,nums[6]).grid_lin()
#        lambd = generateParameter(0.2,0.6,nums[7]).grid_lin()
        alpha, lambd = [0], [0.5]
        return n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, alpha, lambd
    
    else:
        raise Exception('{} has not been implemented yet or len(nums) not right'.format(expert))
        
        
def gen_params_random(nums, expert):
    if expert == "LR" and len(nums)==2:
        alphas=generateParameter(1e-4,1e2,nums[0]).random()
        l1s=generateParameter(0,1,nums[1]).random()
        return alphas,l1s
    
    elif expert == "SVR" and len(nums)==3:
        gammas = generateParameter(1e-4,1e3,nums[0]).random()
        Cs = generateParameter(1e-3,1e4,nums[1]).random()
        epsilons = generateParameter(2**(-6),2**2,nums[2]).random()
        return gammas, Cs, epsilons

    elif expert == "RF" and len(nums)==4:
        n_estimators = generateParameter(100,1000,nums[0]).random("int")
        max_depth = generateParameter(2,15,nums[1]).random("int") 
        min_samples_split = generateParameter(2,15,nums[2]).random("int")
        min_samples_leaf = generateParameter(1,10,nums[3]).random("int")
        return n_estimators, max_depth, min_samples_split, min_samples_leaf

    elif expert == "XGBoost" and len(nums)==8:
        max_depth=generateParameter(2,4,nums[0]).random("int")
        learning_rate=generateParameter(1e-2,1e-1,nums[1]).random()
        n_estimators = generateParameter(100,800,nums[2]).random("int")
        subsample = generateParameter(0.75,1,nums[3]).random()
        colsample_bytree = generateParameter(0.75,1,nums[4]).random()
        gamma = generateParameter(1e-2,1,nums[5]).random()
#        alpha =  generateParameter(0.2,0.6,nums[6]).random()
#        lambd = generateParameter(0.2,0.6,nums[7]).random()
        alpha, lambd = 0, 0.5
        return n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, alpha, lambd
    
    else:
        raise Exception('{} has not been implemented yet or len(nums) not right'.format(expert))

class generateParameter(object):
    '''
    Generate parameter using Grid Search, Random Search,
    and other methods
    '''
    
    def __init__(self,begin, end, n):
        '''
        begin: beginning of the range
        end: end of the range
        n: number of parameters
        '''
        if type(n)!= int:
            print('n needs to be int')
        self.begin = begin
        self.end = end
        self.n = n
        
    def grid_log(self, base=10):
        '''
        grid search
        '''
        return [round(i,4) for i in np.logspace(self.begin,self.end, num=self.n, base=base)]
    
    def grid_lin(self, data_type="float"):
        '''
        grid search
        '''
        if data_type=="float":
            return [round(i,4) for i in np.linspace(self.begin,self.end, num=self.n)]
        elif data_type=="int":
            return [int(i) for i in np.linspace(self.begin,self.end, num=self.n)]
        
    def random(self,data_type="float"):
        if data_type=="float":
            return [round(i,4) for i in np.random.uniform(self.begin, self.end, size=(self.n,))]
        elif data_type=="int":
            return [round(i,4) for i in np.random.randint(self.begin, self.end, size=(self.n,))]
        