#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:39:59 2019

@author: mingmingyu
"""
import math
from dataGen import dataARMA

class trainOL(object):
    '''
    This class train the Online Learner using training data of
    predetermined features
    '''
    def __init__(self,onlineLearner,coefficient,sigma,N,modelName = 'AR',stage=1):
        '''
        input: onlineLearner = the onlineLearner
                coefficient = coefficients of ARMA model used to generate data
                sigma = noise
                N = number of train data 
                stage = number of stages of test data
        '''
        self.onlineLearner = onlineLearner
        self.sigma = sigma
        if stage!=1 and stage!=2:
            print('Warning: stage needs to be either 1 or 2 for now')
            
        if stage==1:        
            inputs,outputs = dataARMA(coefficient,sigma,N*2,model=modelName).generate() #generate data
                    
            # Train the online learner using first half data
            self.onlineLearner.train(outputs[:N]) 
            # The rest of data is test data
            self.inputs=inputs[N:]
            self.outputs=outputs[N:]
            
        else:
            ar_inputs,ar_outputs = dataARMA(coefficient,sigma,N*2).generate()
            ma_inputs,ma_outputs = dataARMA(coefficient,sigma,N*2,model = "MA").generate()
            #Training data
            ar_data_train = ar_outputs[:N]
            ma_data_train = ma_outputs[:N]
            #Test data
            self.inputs = ar_inputs[N:] +  ma_inputs[N:]
            self.outputs = ar_outputs[N:] + ma_outputs[N:]
            
            #Train the model using training data
            for model in self.onlineLearner.models:
                if model.name.startswith('AR'):
                    model.train(ar_data_train)
                elif model.name.startswith('MA'):
                    model.train(ma_data_train)
                else:
                    print('Warning: 2 stage models are not within range')
                                    
    def getTestData(self):
        '''
        Get test data
        '''
        return self.inputs,self.outputs
        
    def getSigma(self):
        return self.sigma
    
    def getLearner(self):
        return self.onlineLearner