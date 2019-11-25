#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from core.online_learner_hpc import *
from utils.plot import *
import numpy as np
import pandas as pd

"""
This is an example to show how do we 
perform online learning using results from hpc

This example is based on the fact that you have already calculate
expert losses and predictions using testing data.

"""
#
#source_path = '/Users/Jiahao/Downloads/output/'
#y_test = np.array(pd.read_csv('/Users/Jiahao/Downloads/test.csv').iloc[:,0])

sigma = 0
p_sigma = 0.5

source_path = '/Users/Jiahao/Documents/classes/capstone/output/{}/'.format(sigma)
y_test = np.array(pd.read_csv("/Users/Jiahao/Documents/classes/capstone/online_learning/data/xgb_test_{}.csv".format(sigma), header=None).iloc[:,0])


# create a online learner calculator
redis = 0.4
learning_rate = 0.5
OL_name= "FTL"
learner = FTL_hpc(source_path = source_path,
                  redis = redis,p_sigma=p_sigma)


# get expert weights change matrix
W = learner.compute_weight_change()

# find the leading expert over time
lead_expert = learner.find_leading_expert(W)

# get algo prediction over time 
P = learner.compute_algo_prediction(W)
# get algo loss over time 
L = learner.compute_algo_loss(P,y_test)

# get algo regret
R,best_expert = learner.compute_algo_regret(L)


# Then we can use our plot utilities
# for example, plot weight change over time

names = learner.model_names
output = '/Users/Jiahao/Downloads/p_sigma{}.png'.format(p_sigma)
#names = ''
plot_weight((W.T),names, title = OL_name+"  "+names[lead_expert[-1]]+"  "+str(redis),output_path = output)












