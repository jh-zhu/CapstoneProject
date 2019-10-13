#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:02:07 2019

@author: team
"""

'''
AR model data
'''
import numpy as np

coeff=np.array([0.5,0.4]) # coefficients
p = len(coeff) # AR(p)
Sigma=1
N=10000
Z=np.random.normal(0,Sigma,N)
X=[0]*N

for i in range(p,N):
    X[i]=coeff.dot(np.array(X[i-p:i]).T) +Z[i] 
plt.plot(X)


'''
MA model data
'''
coeff=np.array([0.5,0.4]) # coefficients
q = len(coeff) # MA(q)
Sigma=1
N=10000
Z=np.random.normal(0,Sigma,N)
X=[0]*N

for i in range(q,N):
    X[i]=coeff.dot(np.array(Z[i-q:i]).T) +Z[i] 
plt.plot(X)