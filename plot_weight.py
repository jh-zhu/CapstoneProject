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


'''
On uniform known data set
'''
# generate ar2 data
phi1,phi2=0.6,0.3 # xt = 0.6x_(t-1) + 0.3 x_(t-2) + Z
sigma=10
N=20000
z=np.random.normal(0,sigma,N)
data_ar2=[0]*N
for i in range(2,N):
    data_ar2[i]=phi1*data_ar2[i-1] + phi2*data_ar2[i-2] + z[i]

# split ar2 generated data
data_ar2_train = data_ar2[:10000]
data_ar2_test = data_ar2[10000:]

# Another way to generate ar2 data
dat=dataGen.dataARMA([0.3,0.6],1,20000)
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
    


# warm up prediction
for i in range(50):
    for ar in ars:
        ar.predict([data_ar2_test[i]])
# put four experts into online learning algorithms
exp1 = exponential_weighted_average(ars,0.05)
exp2 = exponential_weighted_average(ars,0.6)
exp3 = exponential_weighted_average(ars,0.1,redis=0.5)

FTL = follow_the_lead(ars)

OL=FTL # choose online learner
weights = [[]for i in range(n)]

for point in data_ar2_test[50:]:
    OL.update_point([point],[point])
    W = OL.get_weight()
    for i in range(n):
        weights[i].append(W[i])

for weight in weights:
    plt.plot(weight)
plt.legend(['ar1','ar2','ma1','ma2'])
plt.show()

for weight in weights:
    print(sum(weight))


    



    
    




