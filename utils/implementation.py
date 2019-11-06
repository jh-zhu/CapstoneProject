#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 01:02:57 2019

@author: yitongcai
"""

from core.experts import *
from core.choose_experts import *
from core.online_learner import *
from utils.data_generator import *
from utils.testOL import *

'''Create all experts with grid & random methods'''
# first we build an create expert class that our algorithm wants to use with model hyper-parameter
createExperts = experts_create()
# Create experts with random search method 
'''补充parameters'''
createExperts.RandomCreate()          
random_models = createExperts.get_models()
random_modelNames = createExperts.get_modelNames()
# Create experts with grid search method 
createExperts.GridCreate()
grid_models = createExperts.get_models()
grid_modelNames = createExperts.get_modelNames()


'''Create online learner with grid & random experts'''
# then we choose an online learning algorithm we want to use here with hyperparameter redistribution
redis = 0.5
learner = exponential_weighted_average(models,0.05,redis = redis)


'''Get the real dataset with X, y'''
# then we obtain our training and testing data
# normally we will import read fininal data here
# X,y = import(......)


'''Generate data from specific model for training'''
sigmas=np.arange(0,10,1)
N = 1000 # number of data to generate
noise_level = 1 # level of noise (sigma), added to the generating process
kernel = 'linear_regression' # the model to generate 
parameter = [1,[1,2,1.5],noise_level,N] # parameter of data generating model

generator = data_generator(kernel,parameter,noise_level,N)
X_train,y_train = generator.generate()
X_test,y_test = generator.generate()


'''Train & test all experts, and pass tester to plot class'''
# Now we train our experts of our online learning algorithms
learner.train(X_train,y_train)
# Then we test our algorithm with trained experts on testing data and see how our algorithm perform
tester = testOL(learner,X_test,y_test)


'''Plot the experts with given tester'''
plotter =  summary_plots(tester, y_test, sigmas, grid_modelNames)
plotter.plot_weight()
plotter.plot_regret()
plotter.plot_choose_right_expert()


