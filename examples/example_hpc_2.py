#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:12:07 2019

@author: Jiahao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from core.online_learner_hpc import *
from utils.plot import *
import numpy as np
import pandas as pd

"""
This is an example to show how do we 
perform online learning using results from hpc
calculating regret and percent of time choosing the correct expert

This example is based on the fact that you have already calculate
expert losses and predictions using testing data.

"""



loss_file = '/Users/Jiahao/Downloads/loss.xlsx'  # combined loss csv
prediction_file = '/Users/Jiahao/Downloads/prediction.xlsx' # combined predict csv


# read in excel file
# very slow. So read  all sheets here at once
# can do future improvement on speed here
loss_dict = pd.read_excel(loss_file,sheet_name=None,header=0,index_col=0)
prediction_dict = pd.read_excel(prediction_file,sheet_name=None,header=0,index_col=0)


redis = 0
sigmas = [1,5,10,15,20]

regrets = []
for sigma in sigmas:

    y_test = np.array(pd.read_csv('/Users/Jiahao/Documents/classes/capstone/online_learning/data/xgb_test_{}.csv'.format(sigma),header=None).iloc[:,0])
    learner = FTL_hpc(loss_file=loss_dict,prediction_file=prediction_dict,sigma=sigma,redis=redis)
    
    # get expert weights change matrix
    W = learner.compute_weight_change()
    # get algo prediction over time 
    P = learner.compute_algo_prediction(W)
    # get algo loss over time 
    L = learner.compute_algo_loss(P,y_test)
    # get algo regret
    R,best_expert = learner.compute_algo_regret(L)
    
    print('regret {}: {}'.format(sigma,R))
    
    regrets.append(R)


plot_weight(W.T)
plot_regret(sigmas,regrets,title='FTL: regret vs sigma')


percents = []
for sigma in sigmas:

    y_test = np.array(pd.read_csv('/Users/Jiahao/Documents/classes/capstone/online_learning/data/xgb_test_{}.csv'.format(sigma),header=None).iloc[:,0])
    learner = FTL_hpc(loss_file=loss_dict,prediction_file=prediction_dict,sigma=sigma,redis=0.2)
    
    # get expert weights change matrix
    W = learner.compute_weight_change()
    # get algo prediction over time 
    P = learner.compute_algo_prediction(W)
    # get algo loss over time 
    L = learner.compute_algo_loss(P,y_test)
    # get algo regret
    R,best_expert = learner.compute_algo_regret(L)
    
    percent = sum(W[:,best_expert])/len(W)
    
    print('percent choosing right expert {}: {}'.format(sigma,percent))
    
    percents.append(percent)

plot_choose_right_expert(sigmas,percents,title='FTL: percent of choosing right expert')


