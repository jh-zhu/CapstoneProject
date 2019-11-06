#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:06:47 2019

@author: Jiahao

example to use our online learning algorithm
"""

from core.experts import *
from core.online_learner import *
from utils.data_generator import *
from utils.testOL import *


# first we build an expert class that our algorithm wants to use
# with model hyper-parameter
# choose models from core.experts
models = [SARIMAX(1,0,1,0,0,0,12)]#SVR('rbf',0.01,0.5),SVR('linear',0.03,0.1),LinearRegression(),AR(2)]

# then we choose an online learning algorithm we want to use
# Here, use follow the lead for example
# with hyper-parameter redis of our algorithm
redis = 0.5
learner = exponential_weighted_average(models,0.05,redis = redis)

# then we obtain our training and testing data
# normally we will import read fininal data here
# X,y = import(......)

# Here we use our generated toy data, with a specific generating model


N = 1000 # number of data to generate
noise_level = 1 # level of noise (sigma), added to the generating process
kernel = 'linear_regression' # the model to generate 
parameter = [1,[1,2,1.5],noise_level,N] # parameter of data generating model

generator = data_generator(kernel,parameter,noise_level,N)
X_train,y_train = generator.generate()
X_test,y_test = generator.generate()

# Now we train our experts of our online learning algorithms
learner.train(X_train,y_train)

# Then we test our algorithm with trained experts on testing data
# and see how our algorithm perform
tester = testOL(learner,X_test,y_test)

# Then we can use tester to plot weight, compute regret and ...
# In general the tester should be able to give us some feedback of 
# how our algorithm is doing on the test data set

import matplotlib.pyplot as plt
# weight plot
W = tester.compute_weight()
legends = [model.get_name() for model in learner.models]

fig = plt.figure(figsize = (12,4))
for w in W:
    _ = plt.plot(w)
plt.show()















