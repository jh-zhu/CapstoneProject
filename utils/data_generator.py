#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:58:14 2019

@author: yitongcai
"""

import numpy as np
import statsmodels.tsa.arima_process as tsa


class ARMA_generator(object):
    '''
    Generate ARMA model data
    '''
    def __init__(self, ar_coefs, ma_coefs, sigma, N):
        
        self.ar_coefs= ar_coefs # coefficients list in AR
        self.ma_coefs= ma_coefs # coefficients list in MA
        self.sigma = sigma # volatility of the noise
        self.N = N # number of data point
                
    def generate(self):
        '''
        Generate data using pre-determined coefficients
        input: [[x1,z1],[x2,z2],......[xn,zn]]
        output: [y1,y2,y3,.....,yn]
        '''
        np.random.seed(12345)
        arparams = np.array(self.ar_coefs)
        maparams = np.array(self.ma_coefs)
        print(arparams,maparams)
        ar = np.r_[1, -arparams] # add zero-lag and negate
        ma = np.r_[1, maparams] # add zero-lag
        outputs = tsa.arma_generate_sample(ar, ma, self.N)
        return outputs
    