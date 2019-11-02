#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:58:14 2019

@author: yitongcai
"""

import numpy as np
import statsmodels.tsa.arima_process as tsa
import sys


class data_generator(object):
    '''
    Data generator that generate data with specific core model
    Output data format: 
        y:[N]
        X: [N*[P]] N*P matrix, rows as sample number, columns as feature numbers
        
        If it is 1-d time series data 
        y:[N]
        X:[[y]]
    '''
    
    
    def __init__(self,kernel,parameter,sigma,N):
        '''
        input: 
            kernel: 'linear_regression', 'ARMA'
            params: '[alpha,[beta1,beta2,...,betap]]'
                    or '[[ar-coefficients],[ma-coefficients]]'
            sigma: noise level
            N:  number of points to generate 
        '''
        
        if kernel == 'linear_regression':
            self.kernel = linear_regression_generator(parameter,sigma,N)
        elif kernel == 'ARMA':
            self.kernel = ARMA_generator(parameter,sigma,N)
        else:
            print('This generating kernel has not yet implemented')
            sys.exit(1)
        
    def generate(self):
        return self.kernel.generate()


class ARMA_generator(object):
    '''
    Generate ARMA model data
    '''
    def __init__(self, params, sigma, N):
        
        self.ar_coefs= params[0] # coefficients list in AR
        self.ma_coefs= params[1] # coefficients list in MA
        self.sigma = sigma # volatility of the noise
        self.N = N # number of data point
                
    def generate(self):
        '''
        Generate data using pre-determined coefficients
        output: 
            X:  None
            y: [N]
        '''
        #np.random.seed(12345)
        ar = np.r_[1, [-i for i in self.ar_coefs]]  # add zero-lag and negate
        ma = np.r_[1, self.ma_coefs]                # add zero-lag
        outputs = tsa.arma_generate_sample(ar, ma, self.N).tolist()
        return None,outputs


class linear_regression_generator(object):
    
    '''
    Generate data using linear_regression_model
    '''
    
    def __init__(self,params,sigma,N):
        self.alpha = params[0]
        self.betas = params[1]
        self.sigma = sigma
        self.N = N
        
    
    def generate(self):
        # first generate the design matrix X:  n*p
        dimension = len(self.betas) # the dimension of design matrix, ie p
        # the scale of X can be vary, this can be improved later on 
        X = np.array([np.random.normal(0,10,self.N) for i in range(dimension)]).T
        
        # then  compute y 
        y = np.array([self.alpha]*self.N)
        for i in range(dimension):
            y = y + self.betas[i]*X[:,i]
        y = y + np.random.normal(0,self.sigma,self.N)
        
        return X.tolist(), y.tolist()
        
        
        
        
        
        
        
        
        
        
        
        
        
        


    



        
        
        
        
        
        
        
    