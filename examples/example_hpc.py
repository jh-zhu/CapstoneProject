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
sigma = 5

#source_path = '/Users/Jiahao/Documents/classes/capstone/output/{}/'.format(sigma)
#y_test = np.array(pd.read_csv("/Users/Jiahao/Documents/classes/capstone/online_learning/data/xgb_test_{}.csv".format(sigma), header=None).iloc[:,0])

source_path = '/Users/yitongcai/Coding/output/{}/'.format(sigma)
y_test = np.array(pd.read_csv("/Users/yitongcai/Coding/data/xgb_test_{}.csv".format(sigma), header=None).iloc[:,0])
output = None

# create a online learner calculator
redis = 0
learning_rate = 0.5
OL_name= "RWM"
learner = RWM_hpc(source_path = source_path,
                  learning_rate = learning_rate,
                  redis = redis)


# get expert weights change matrix
#W = learner.compute_weight_change()
W = learner.compute_underlying_weight_change()

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
#output = '/Users/Jiahao/Downloads/plot3/sigma{}.png'.format(sigma)
#names = ''
fig = plot_weight((W.T),names, title = OL_name+"  "+names[lead_expert[-1]]+"  "+str(redis)+"  "+str(sigma)
#         +" with threshold"
         , output_path = output)



#from matplotlib.backends.backend_pdf import PdfPages
#pdf_pages = PdfPages('/Users/yitongcai/Graduate/NYU研究生/3rd Semsester/Project & Presentation/fig.pdf')

#pdf_pages.savefig(fig2)




