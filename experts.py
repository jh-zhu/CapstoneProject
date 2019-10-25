#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:27:17 2019

@author: yitongcai
"""
from pmdarima import auto_arima
import numpy as np
import statsmodels.tsa.statespace.sarimax as sarimax


class ARIMA(object):
    '''
    ARIMA(p,d,q) model, first trained, and then used to make prediction
    '''
    def __init__(self,p,d,q,freq):
        '''
        p: max order of AR model
        q: max order of MA model
        freq: ex. daily:365 / quarter:4 / month:12 / annual:1
        '''
        self.p=p 
        self.d=d
        self.q=q
        self.freq=freq
        self.train_data=None #training data
        self.data=None #saved data used to make prediction
        self.coeff=None
        self.name = 'ARMA({},{},{})'.format(self.p, self.d, self.q)
        
    def get_name(self):
        return self.name

        
    def train(self,train_data):
        '''
        Train the data using train_data
        '''    
        self.train_data=train_data
        self.data=train_data[-self.q:]
        # m refers to the number of periods in each season   
        stepwise_model = auto_arima(self.train_data,start_p=0, start_q=0,
                                       max_p=self.p, max_q=self.q, m=self.freq,
                                        d=self.d, D=None, trace=True,
#                                       error_action='ignore',  
#                                      suppress_warnings=True
                                        stepwise=True)
#        print( stepwise_model.summary())
        order = stepwise_model.get_params()['order']
        seasonal_order = stepwise_model.get_params()['seasonal_order']
        model=sarimax.SARIMAX(endog=train_data,order=order,seasonal_order=seasonal_order,trend='c',enforce_invertibility=False)
        results=model.fit()
#        print(results.summary())
        '''constant ARs MAs variance_of_error_term'''
        self.coeff = results.params[:-1]

   
    def predict(self,test_data):
        '''
        pass one point and make one prediction
        '''
        # It's a AR model, get the first argument (x)
        stepwise_model.predict()
#        new_x = test_data[0]
#        
#        res = np.flip(np.array(self.coeff[:-1])).dot(np.array(self.data[-self.p:]).T)
#        self.data.append(new_x)
#        self.data.pop(0)
        return res
