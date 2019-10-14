#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 17:48:25 2019

@author: Ming,Jiahao
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import math


class AR(object):
    '''
    AR(p) model, first trained, and then used to make prediction
    '''
    def __init__(self,p):
        self.p=p 
        self.train_data=None #training data
        self.data=None #saved data used to make prediction
        self.coeff=None
        self.name = 'AR{}'.format(self.p)
        
    def train(self,train_data):
        '''
        Train the data using train_data
        '''
        self.train_data=train_data
        self.data=train_data[-self.p:]
        initials=[0.1]*(self.p + 1) # p phi and 1 sigma
        
        self.coeff=fmin(self.MLE,initials)

    def MLE(self,initial):
        '''
        MLE used to solve the coefficients of AR(p) model
        '''
    
        Phi,Sigma=np.array(initial[:-1]),initial[-1]
        N = len(self.train_data)
        Z=[0]*N
        Z[0]=0
        Summation = 0
        for i in range(self.p,N):
            'X is from data initialization'
            Z[i]=self.train_data[i]-Phi.dot(np.array(self.train_data[i-self.p:i]).T)
            Summation += -1*((Z[i]**2)/(2*Sigma**2))
        res=(-1*(N-1)/2) * np.log(2*math.pi)-((N-1)/2) * np.log(Sigma **2) + Summation
        return -res
    
    def predict(self,test_data):
        '''
        pass one point and make one prediction
        '''
        if len(test_data)>1: print('pass one point only')
        res = np.array(self.coeff[:-1]).dot(np.array(self.data[-self.p:]).T)
        self.data.extend(test_data)
        self.data.pop(0)
        return res
    
    def get_name(self):
        return self.name
    
class MA(object):
    '''
    MA(q) model, first trained, and then used to make prediction
    '''
    def __init__(self,q):
        self.q=q 
        self.train_data=None #training data
        self.data=None #saved data used to make prediction
        self.coeff=None
        self.name = 'MA{}'.format(self.q)
        
    def train(self,train_data):
        '''
        Train the data using train_data
        '''
        self.train_data=train_data
        self.data=train_data[-self.q:]
        initials=[0.1]*(self.q + 1) # p phi and 1 sigma
        
        self.coeff=fmin(self.MLE,initials)

    def MLE(self,initial):
        '''
        MLE used to solve the coefficients of MA(q) model
        '''
    
        Theta,Sigma=np.array(initial[:-1]),initial[-1]
        N = len(self.train_data)
        Z=[0]*N
        Z[0]=0
        Summation = 0
        for i in range(self.q,N):
            'X is from data initialization'
            Z[i]=self.train_data[i]-Theta.dot(np.array(Z[i-self.q:i]).T) 
            Summation += -1*((Z[i]**2)/(2*Sigma**2))
        res=(-1*(N-1)/2) * np.log(2*math.pi)-((N-1)/2) * np.log(Sigma **2) + Summation
        return -res
    
    def predict(self,test_data):
        '''
        pass one point and make one prediction
        '''
        if len(test_data)>1: print('pass one point only')
        res=np.array(self.coeff[:-1]).dot(np.array(self.data[-self.q:]).T)
        self.data.extend(test_data)
        self.data.pop(0)
        return res
    
    def get_name(self):
        return self.name
    
