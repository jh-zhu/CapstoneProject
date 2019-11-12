#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from core.online_learner_hpc import *
from utils import plot
import numpy as np

"""
This is an example to show how do we 
perform online learning using results from hpc

This example is based on the fact that you have already calculate
expert losses and predictions using testing data.

"""

source_path = '/Users/Jiahao/Downloads/output/'


# create a online learner calculator
redis = 0.5
learner = FTL_hpc(source_path,redis = redis)


# get expert weights change matrix
W = learner.compute_weight_change()

# get algo prediction over time 
P = learner.compute_algo_prediction()

# get algo loss over time 
L = learner.compute_algo_loss(y_test)

# get algo 
R = learner.compute_algo_regret(y_test)


# Then we can use our plot utilities
# for example, plot weight change over time

names = learner.model_names
plot.plot_weight(np.array(W).T,names)












