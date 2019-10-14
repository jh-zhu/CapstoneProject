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
    def __init__(self,onlineLearner,coefficient,sigma,N,modelName = 'AR',stage=1,redistribute=0):
        '''
        input: onlineLearner = the onlineLearner
                coefficient = coefficients of ARMA model used to generate data
                sigma = noise
                N = number of train data 
                stage = number of stages of test data
                redistribute = redistribute loss
        '''
        self.onlineLearner = onlineLearner
        self.dataGenerator = dataARMA(coefficient,sigma,N*2,model=modelName) #generate first stage data
        data=self.dataGenerator.generate() #generate data using model (AR or MA)
                
        # split generated data to train data and test data
        self.trainData = data[:N] #half amount of data is used to train the online learner
        self.onlineLearner.train(self.trainData) # train the online learner
        self.testData=None
        if stage==1:
            self.testData = data[N:]
            
        elif stage==2:
            self.dataGenerator2 = dataARMA(coefficient,sigma,math.floor(N/2),model="MA")
            testData2=self.dataGenerator2.generate()
            self.testData = data[N:math.floor(1.5*N)] + testData2
        else:
            print('stage needs to be less than 2 for now')

        self.sigma = sigma
        
    def getTestData(self):
        '''
        Get test data
        '''
        return self.testData
        
    def getSigma(self):
        return self.sigma
    
    def getLearner(self):
        return self.onlineLearner