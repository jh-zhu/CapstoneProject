#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:02:07 2019

@author: team
"""


import numpy as np

class dataARMA(object):
    '''
    Generate AR or MA model data
    '''
    def __init__(self, coeff, sigma, N, model='AR'):
        
        self.coeff = coeff #phi in AR, or theta in MA
        self.sigma = sigma #volatility of the noise
        self.N = N #number of data point
        self.model = model #AR or MA data
                
    def generate(self):
        '''
        Generate data using pre-determined coefficients
        output: generated data
        '''
        X=[0.01]*self.N
        p = len(self.coeff)
        Z=np.random.normal(0,self.sigma,self.N)

        if self.model!='AR' and self.model!='MA':
            print(f'model is {self.model}')
            return None
            
        for i in range(p,self.N):
            if self.model=='AR':
                X[i]=np.array(self.coeff).dot(np.array(X[i-p:i]).T) + Z[i]
            else:
                X[i]=np.array(self.coeff).dot(np.array(Z[i-p:i]).T) + Z[i]             
        return X
