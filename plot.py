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





# models 
ar1 = AR(1)
ar2 = AR(2)
ma1 = MA(1)
ma2 = MA(2)
models = [ar1,ar2,ma1,ma2]

# online learner
learner = exponential_weighted_average(models,0.05)

# trainning 
coefficients = [0.3,0.4]
sigma = 1
N = 1000
trainer = trainOL(learner,coefficients,sigma,N,stage=2)

test_data = trainer.getTestData()

# test 
tester = testOL(learner,test_data)

# weight plot
tester.weight_plot()

#########################################################
# plot regret over noise 
sigmas = np.arange(0,10,1)
regrets = []

for sigma in sigmas:
    learner = exponential_weighted_average(models,0.05)
    trainer = trainOL(learner,coefficients,sigma,N)
    test_data = trainer.getTestData()
    
    tester = testOL(learner,test_data)
    regret = tester.compute_regret()
    regrets.append(regret)


plt.plot(sigmas,regrets)
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
two stages 
'''

N = 2000

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
    

learner = exponential_weighted_average([ar1,ar2,ma1,ma2],0.05)

tester = testOL(learner,data_test)

tester.weight_plot()












    
    
    






