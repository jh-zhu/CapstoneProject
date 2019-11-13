#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from core.experts import *
from core.online_learner import *
from utils.data_generator import *
from utils.testOL import *
from utils import plot

"""

Example to use our online-earning algorithm.

In this example, we show how to calculate weight change 
of experts over time. 

"""



# first we build an expert class that our algorithm wants to use
# with model hyper-parameter
# choose models from core.experts
models = [SVR('linear',0.01,0.5),SVR('linear',0.03,0.1),LinearRegression(0.0001,0.1),AR(2)]


# then we choose an online learning algorithm we want to use
# Here, use exponential weighted average for example
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
parameter = [1,[1,2,1.5]] # parameter of data generating model

generator = data_generator(kernel,parameter,noise_level,N)
X_train,y_train = generator.generate()
X_test,y_test = generator.generate()

# Now we train our experts of our online learning algorithms
learner.train(X_train,y_train)

# Then we test our algorithm with trained experts on testing data
# and see how our algorithm perform
tester = testOL(learner,X_test,y_test)

# we can get the weight change matrix W from tester
W = tester.compute_weight()

# plot and show results using plot utilities

model_names = [model.get_name() for model in models]
size = (12,4)
title = 'expert weights change'
#output_path = '/Users/Jiahao/Documents/weight.png'

plot.plot_weight(W,model_names,title,size)

















