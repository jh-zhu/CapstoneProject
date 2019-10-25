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
        input: [[x1,z1],[x2,z2],......[xn,zn]]
        output: [y1,y2,y3,.....,yn]
        '''
        X=[0.01]*self.N
        p = len(self.coeff)
        Z=np.random.normal(0,self.sigma,self.N)
        inputs=[[0]*2]* self.N
        outputs=[0]* self.N

        if self.model!='AR' and self.model!='MA':
            print(f'model is {self.model}')
            return None
                
        for i in range(p,self.N):
            if self.model=='AR':
                X[i]=np.array(self.coeff).dot(np.array(X[i-p:i]).T) + Z[i]
            else:
                X[i]=np.array(self.coeff).dot(np.array(Z[i-p:i]).T) + Z[i]
            inputs[i]=[X[i],Z[i]]
            outputs[i]=X[i]
        return inputs,outputs
        
        '''
        for i in range(p,self.N):
            X[i]=np.array(self.coeff).dot(np.array(X[i-p:i]).T) + Z[i]
            Z[i]=np.array(self.coeff).dot(np.array(Z[i-p:i]).T)
            inputs[i] = [X[i],Z[i]]
            if self.model == "AR":
                outputs[i] = X[i]
            else:
                outputs[i] = Z[i]
        # add first few items
        for j in range(0,p):
            inputs[j] = [X[j],Z[j]]
            if self.model == "AR":
                outputs[j] = X[j]
            else:
                outputs[j] = Z[j]
        '''
        
        return inputs,outputs
