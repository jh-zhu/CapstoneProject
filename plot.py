#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:29:55 2019

@author: Jiahao
"""

src_path = '/Users/Jiahao/Documents/classes/capstone/online_learning/'
import os
os.chdir(src_path)

from online_learner import *
from ARMA import *
from trainOL import *
from testOL import * 
import numpy as np
import matplotlib.pyplot as plt

from dataGen import *




############################################################
# stage1 or stage 3, depends on the length of coefficient
# models 
ar1 = AR(1)
ar2 = AR(2)
ma1 = MA(1)
ma2 = MA(2)
models = [ar1,ar2,ma1,ma2]

# online learner
redis = 0
learner = exponential_weighted_average(models,0.05,redis=redis)

# trainning 
coefficients = [0.3,0.4,0.6]
sigma = 1
N = 3000
trainer = trainOL(learner,coefficients,sigma,N,modelName='MA')

test_data = trainer.getTestData()

# test 
tester = testOL(learner,test_data)

# weight plot
title  = "weight_EWA_stage3_{}_{}".format(redis,sigma)
xlabel = "data point"
ylabel = "weight"
tester.weight_plot(title,xlabel,ylabel)




#########################################################
# plot regret over noise 
sigmas = np.arange(0,10,1)
regrets = []
redis = 0.3
coefficients = [0.3,0.4,0.6]

for sigma in sigmas:
    learner = exponential_weighted_average(models,0.05,redis=redis)
    trainer = trainOL(learner,coefficients,sigma,N,modelName='MA')
    test_data = trainer.getTestData()
    
    tester = testOL(learner,test_data)
    regret = tester.compute_regret()
    regrets.append(regret)


title = "regret_EWA_stage3_{}".format(redis)
xlabel = 'sigma'
ylabel = 'regret'

plt.scatter(sigmas,regrets)
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()


##########################################################
# plot number of time choosing right expert over noise 
sigmas = np.arange(0,10,1)
percents = []

for sigma in sigmas:
    learner = follow_the_lead(models)
    trainer = trainOL(learner,coefficients,sigma,N)
    test_data = trainer.getTestData()
    
    tester = testOL(learner,test_data)
    percent = tester.compute_choose_right_expert()
    percents.append(percent)


plt.plot(sigmas,percents)
plt.show()


learner = follow_the_lead(models)
trainer = trainOL(learner,coefficients,30,N)
test_data = trainer.getTestData()
    
tester = testOL(learner,test_data)
percent = tester.compute_choose_right_expert()






###########################################3
'''
Stage2, which is more complicated. The way you train model, and get test_date
is different.

This code is for regret computation. If you are doing weight stage2, the easiest way 
is put sigma as a array of length 1 and call tester.weight_plot
'''
N = 2000
coefficients = [0.3,0.4]
redis = 0.3
sigmas = np.arange(0,10,1)

regrets = []
for sigma in sigmas:

    ar_data = dataARMA(coefficients,sigma,N).generate()
    ma_data = dataARMA(coefficients,sigma,N,model = "MA").generate()

    ar_data_train = ar_data[:1000]
    ma_data_train = ma_data[:1000]

    ar_data_test = ar_data[1000:]
    ma_data_test = ma_data[1000:]

    data_test = ar_data_test + ma_data_test 

    ar1 = AR(1)
    ar2 = AR(2)
    ma1 = MA(1)
    ma2 = MA(2)

    for ar in [ar1,ar2]:
        ar.train(ar_data_train)

    for ma in [ma1,ma2]:
        ma.train(ma_data_train)
    

    learner = exponential_weighted_average([ar1,ar2,ma1,ma2],0.05,redis=redis)
    tester = testOL(learner,data_test)

    regret = tester.compute_regret()
    regrets.append(regret)
    

title = "regret_EWA_stage2_{}".format(redis)
xlabel = 'sigma'
ylabel = 'regret'

plt.scatter(sigmas,regrets)
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()













    
    
    






