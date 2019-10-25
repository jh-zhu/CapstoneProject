#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:04:25 2019

@author: Jiahao

This file generates weight change plots over time of expert
"""

src_path = '/Users/Jiahao/Documents/classes/capstone/online_learning/'
import os
os.chdir(src_path)

from online_learner import *
from ARMA import *
import numpy as np
import matplotlib.pyplot as plt
import dataGen 

from testOL import *


'''
On uniform known data set
'''


# Another way to generate ar2 data
dat=dataGen.dataARMA([0.3,0.6],10,20000)
data=dat.generate()
data_ar2_train = data[:10000]
data_ar2_test = data[10000:]

# build ar 1,2,3,4 four models, and train those models using the first 10000 data
n = 4 # number of experts
ar1 = AR(1)
ar2 = AR(2)
ar3 = MA(1)
ar4 = MA(2)
ars = [ar1,ar2,ar3,ar4]
for ar in ars:
    ar.train(data_ar2_train)
    


# put four experts into online learning algorithms
exp1 = exponential_weighted_average(ars,0.05)
#exp2 = follow_the_lead(ars,0.05)


exp1.update_point([data_ar2_test[0]],[data_ar2_test[0]])

test = testOL(exp1,data_ar2_test)

test.weight_plot()

percent = test.compute_choose_right_expert()



exp1.models[0].get_name()




    



    
    




