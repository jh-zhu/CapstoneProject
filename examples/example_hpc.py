#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from core.online_learner_calculator import *
from core.utils import plot

"""
This is an example to show how do we 
perform online learning using results from hpc

This example is based on the fact that you have already calculate
expert losses and predictions using testing data.

"""

loss_file_path = '/Users/Jiahao/Downloads/losses.csv'
prediction_file_path = '/Users/Jiahao/Downloads/predictions.csv'


# create a online learner calculator
redis = 0.5
calculator = EWA_calculator(0.05,redis = redis,loss_file_path,prediction_file_path)

# get expert weights change matrix
W = calculator.compute_weight_change()

# get algo prediction over time 
P = calculator.compute_algo_prediction()

# get algo loss over time 
L = calculator.compute_algo_loss(y_test)

# get algo 
R = calculator.compute_algo_regret(y_test)

# get algo percent of time choosing right expert
P = calculator.compute_algo_percent()


# Then we can use our plot utilities
# for example, plot weight change over time

names = calculator.get_expert_names()
plot.plot_weight(W,model_names,title,size,output_path)












