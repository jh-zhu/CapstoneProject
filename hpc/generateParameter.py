#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:26:11 2019

@author: mingmingyu
"""

import numpy as np

class generateParameter(object):
    '''
    Generate parameter using Grid Search, Random Search,
    and other methods
    '''
    
    def __init__(self,begin,end,n):
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
        
    def grid(self):
        '''
        grid search
        '''
        return [round(i,2) for i in np.arange(self.begin,self.end,(self.end-self.begin)/self.n)]
    
    def random(self):
        '''
        random search
        '''
        return np.random.uniform(self.begin,self.end,self.n)
    