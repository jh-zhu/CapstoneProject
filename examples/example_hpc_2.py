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

#source_path = '/Users/yitongcai/Coding/output/{}/'.format(sigma)
#y_test = np.array(pd.read_csv("/Users/yitongcai/Coding/data/xgb_test_{}.csv".format(sigma), header=None).iloc[:,0])


learning_rate = 0.5
redis = 0
sigmas = [0,1,5,10,15,20]

regrets = []
for sigma in sigmas:
#    source_path = '/Users/Jiahao/Documents/classes/capstone/output/{}/'.format(sigma)
#    y_test = np.array(pd.read_csv('/Users/Jiahao/Documents/classes/capstone/online_learning/data/xgb_test_{}.csv'.format(sigma),header=None).iloc[:,0])

    source_path = '/Users/yitongcai/Coding/output/{}/'.format(sigma)
    y_test = np.array(pd.read_csv("/Users/yitongcai/Coding/data/xgb_test_{}.csv".format(sigma), header=None).iloc[:,0])
  
    
    learner = RWM_hpc(source_path = source_path, learning_rate = learning_rate, redis=redis)
    
    # get expert weights change matrix
    W = learner.compute_underlying_weight_change()
    # get algo prediction over time 
    P = learner.compute_algo_prediction(W)
    # get algo loss over time 
    L = learner.compute_algo_loss(P,y_test)
    # get algo regret
    R,best_expert = learner.compute_algo_regret(L)
    
    print('regret {}: {}'.format(sigma,R))
    
    regrets.append(R)

fig5 = plot_regret(sigmas,regrets,title='RWM: regret vs sigma'+" redis: "+str(redis))
#pdf_pages.savefig(fig5)


percents = []
for sigma in sigmas:
#    source_path = '/Users/Jiahao/Documents/classes/capstone/output/{}/'.format(sigma)
#    y_test = np.array(pd.read_csv('/Users/Jiahao/Documents/classes/capstone/online_learning/data/xgb_test_{}.csv'.format(sigma),header=None).iloc[:,0])
    
    source_path = '/Users/yitongcai/Coding/output/{}/'.format(sigma)
    y_test = np.array(pd.read_csv("/Users/yitongcai/Coding/data/xgb_test_{}.csv".format(sigma), header=None).iloc[:,0])
  
    learner = RWM_hpc(source_path = source_path, learning_rate = learning_rate, redis=redis)
    
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

fig5 = plot_choose_right_expert(sigmas,percents,title='RWM: percent of choosing right expert'+" redis: "+str(redis))
#pdf_pages.savefig(fig5)
#
#
#pdf_pages.close()


