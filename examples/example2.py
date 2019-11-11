#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from core.experts import *
from core.online_learner import *
from utils.data_generator import *
from utils.testOL import *
from utils import plot
import numpy as np


"""
Example to use our online-earning algorithm.

In this example, we show how to calculate 

1. noise vs regret 

2. noise vs percent of time choosing right expert
"""

# create models from expert class
models = [SVR('linear',0.01,0.5),SVR('linear',0.03,0.1),LinearRegression(),AR(2)]

# then we choose an online learning algorithm we want to use
# Here, use exponential weighted average for example
# with hyper-parameter redis of our algorithm
redis = 0.5
learner = exponential_weighted_average(models,0.05,redis = redis)



# now we want to create a list of sigmas (data noise), and corresponding regret
sigmas = np.arange(0,1,0.2)
regrets = np.zeros(len(sigmas))

N = 1000 
kernel = 'linear_regression' 
parameter = [1,[1,2,1.5]]

# now we create data and test
for i,sigma in enumerate(sigmas):
    generator = data_generator(kernel,parameter,sigma,N)
    X_train,y_train = generator.generate()
    X_test,y_test = generator.generate()
    
    learner.train(X_train,y_train)
    tester = testOL(learner,X_test,y_test)
    
    regret = tester.compute_regret()
    regrets[i] = regret

# now we plot regrets against sigma 
size = (12,4)
title = 'algo regret vs noise'
output_path = '/Users/Jiahao/Documents/regret.png'

plot.plot_regret(sigmas,regrets,title,size,output_path)
    
    







