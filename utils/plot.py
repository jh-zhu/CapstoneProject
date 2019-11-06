#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jiahao, Yitong
"""
from core.online_learner import *
from depreciated.ARMA import *
from depreciated.trainOL import *
from utils.testOL import * 
from utils.data_generator import *
from core.experts import *

import numpy as np
import matplotlib.pyplot as plt
import os
#src_path = '/Users/Jiahao/Documents/classes/capstone/online_learning/'
#src_path = '/Users/yitongcai/Coding/CapstoneProject/'
#src_path='/scratch/mmy272/test/CapstoneProject/'
#os.chdir(src_path)

class summary_plots(object):
    def __init__(self, tester, y_test, sigmas, model_names):
        self.tester = tester  
        self.y_test = y_test
        self.sigmas = sigmas
        self.model_names = model_names
        
        
    def plot_weight(self):
        '''
        plot weight for all the experts over data inputs
        '''
        weights = self.tester.compute_weight()      
        experts = self.tester.online_learner.models
        names = []
        for expert in experts:
            names.append(expert.get_name())       
        for weight in weights:
            plt.plot(weight[:len(self.y_test)])
            
#        title  = "weight_{}_stage{}_{}_{}".format(self.learner_name, stage, redis, sigma)
#        title  = "stability : {}".format( redis)
        title=''        
        xlabel = "data point"
        ylabel = "weight"
        plt.legend(names)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
        
    def plot_regret(self):
        '''
        plot regret over noise 
        '''
        regrets = []
        for sigma in self.sigmas:
            regret = self.tester.compute_regret()
            regrets.append(regret)
 
#        title  = "regret_{}_stage{}_{}_{}".format(self.learner_name, stage, redis, sigma)
        title = " "
        xlabel = 'sigma'
        ylabel = 'regret'
        plt.scatter(self.sigmas,regrets)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        

    def plot_choose_right_expert(self):
        '''
        plot number of time choosing right expert over noise 
        '''
        percents, indexes = [], []       
        for sigma in self.sigmas:
            index, percent = self.tester.compute_choose_right_expert()
            percents.append(percent)
            indexes.append(index)
        
#        title  = "best expert_{}_stage{}_{}".format(self.learner_name, stage, redis)
        title = " "
        xlabel = 'sigma'
        ylabel = 'percent'
 
        plt.plot(self.sigmas, percents)
        for i in range(len(self.sigmas)):
            plt.annotate("({},{})".format(self.model_names[indexes[i]],percents[i]), 
                                          xy=(self.sigmas[i],percents[i]), xytext=(-20, 10), 
                                          textcoords='offset points')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        

