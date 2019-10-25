#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jiahao, Yitong
"""
from online_learner import *
from ARMA import *
from trainOL import *
from testOL import * 
from dataGen import *
from experts import *

import numpy as np
import matplotlib.pyplot as plt
import os
#src_path = '/Users/Jiahao/Documents/classes/capstone/online_learning/'
src_path = '/Users/yitongcai/Coding/CapstoneProject/'
os.chdir(src_path)

class summary_plots(object):
    def __init__(self, learner_name):
        # models 
        self.learner_name = learner_name
#        ar1 = AR(1)
#        ar2 = AR(2)
#        ma1 = MA(1)
#        ma2 = MA(2)
#        self.models = [ar1,ar2,ma1,ma2]
#        self.model_names=[ar1.name,ar2.name,ma1.name,ma2.name]
        
        '''ARIMA'''
        arima = ARIMA(2,0,2,1)
        self.models = [arima]
        self.model_names=[arima.name]
        
         
        
    def plot_weight(self, redis, sigma, coefficients, N, stage):
        
        if self.learner_name == "EWA":
            learner = exponential_weighted_average(self.models,0.01,redis=redis)
        elif self.learner_name == "RWM":
            learner = randomized_weighted_majority(self.models,0.5,redis=redis)
        elif self.learner_name == "FTL":
            learner = follow_the_lead(self.models)

        # In stage 1, the data is generated by AR(2)
        if stage==1 or stage==2:
            trainer = trainOL(learner,coefficients,sigma,N,modelName = 'AR',stage=stage)
        # In stage 3, the data is generated by MA(3)
        elif stage==3:
            trainer = trainOL(learner,coefficients,sigma,N,modelName='MA',stage=1)
        test_data = trainer.getTestData()       
        tester = testOL(learner,test_data[0],test_data[1])
            
#        title  = "weight_{}_stage{}_{}_{}".format(self.learner_name, stage, redis, sigma)
#        title  = "stability : {}".format( redis)
        title=''
        
        xlabel = "data point"
        ylabel = "weight"
        tester.weight_plot(title,xlabel,ylabel)
        
        
    def plot_regret(self, redis, sigmas, coefficients, N, stage):
        '''
        plot regret over noise 
        '''
        if self.learner_name == "EWA":
            learner = exponential_weighted_average(self.models,0.01,redis=redis)
        elif self.learner_name == "RWM":
            learner = randomized_weighted_majority(self.models,0.5,redis=redis)
        elif self.learner_name == "FTL":
            learner = follow_the_lead(self.models)
            
        regrets = []
        for sigma in sigmas:
            # In stage 1, the data is generated by AR(2)
            if stage==1 or stage==2:
                trainer = trainOL(learner,coefficients,sigma,N,modelName = 'AR',stage=stage)
            # In stage 3, the data is generated by MA(3)
            elif stage==3:
                trainer = trainOL(learner,coefficients,sigma,N,modelName='MA',stage=1)
            test_data = trainer.getTestData()       
            tester = testOL(learner,test_data[0],test_data[1])
            
            regret = tester.compute_regret()
            regrets.append(regret)
 
        title  = "regret_{}_stage{}_{}_{}".format(self.learner_name, stage, redis, sigma)
        xlabel = 'sigma'
        ylabel = 'regret'        
        plt.scatter(sigmas,regrets)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_choose_right_expert(self, redis, sigmas, coefficients, N, stage):
        '''
        plot number of time choosing right expert over noise 
        '''
        percents, indexes = [], []
        if self.learner_name == "RWM":
            learner = randomized_weighted_majority(self.models,0.5,redis=redis)
        elif self.learner_name == "FTL":
            learner = follow_the_lead(self.models)
        
        for sigma in sigmas:
            trainer = trainOL(learner,coefficients,sigma,N)
            test_data = trainer.getTestData()
            tester = testOL(learner,test_data[0],test_data[1])
            index, percent = tester.compute_choose_right_expert()
            percents.append(percent)
            indexes.append(index)
        
        title  = "best expert_{}_stage{}_{}".format(self.learner_name, stage, redis)
        xlabel = 'sigma'
        ylabel = 'percent'
 
        plt.plot(sigmas,percents)
        for i in range(len(sigmas)):
            plt.annotate("({},{})".format(self.model_names[indexes[i]],percents[i]), xy=(sigmas[i],percents[i]), xytext=(-20, 10), textcoords='offset points')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        



sigmas=np.arange(0,10,1)
N=250

summary_plots = summary_plots("FTL")
summary_plots.plot_weight(redis=0, sigma=1, coefficients = [0.75, -0.25, 0.65, 0.35], N=N, stage=1)
#summary_plots.plot_weight(redis=0, sigma=1, coefficients = [0.3,0.4,0.6], N=N, stage=3) #有问题
#summary_plots.plot_weight(redis=0, sigma=1, coefficients = [0.3,0.4], N=N, stage=2)
#summary_plots.plot_weight(redis=0.5, sigma=1, coefficients = [0.3,0.4], N=N, stage=2)
#summary_plots.plot_weight(redis=0.8, sigma=1, coefficients = [0.3,0.4], N=N, stage=2)
#summary_plots.plot_weight(redis=1, sigma=1, coefficients = [0.3,0.4], N=N, stage=2)
#
#summary_plots.plot_regret(redis=0, sigmas=sigmas, coefficients=[0.3,0.4], N=N, stage=1)
#summary_plots.plot_regret(redis=0, sigmas=sigmas, coefficients=[0.3,0.4,0.6], N=N, stage=3)
#summary_plots.plot_regret(redis=0, sigmas=sigmas, coefficients=[0.3,0.4], N=N, stage=2) #有问题
#
#summary_plots.plot_choose_right_expert(redis=0, sigmas=sigmas, coefficients=[0.3,0.4], N=N, stage=1)
#summary_plots.plot_choose_right_expert(redis=0, sigmas=sigmas, coefficients=[0.3,0.4,0.6], N=N, stage=3)
#summary_plots.plot_choose_right_expert(redis=0, sigmas=sigmas, coefficients=[0.3,0.4], N=N, stage=2)

