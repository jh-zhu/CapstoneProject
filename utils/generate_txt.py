#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:28:32 2019

@author: Jiahao
"""

import os

# number of rounds
N = 100
n_rounds = [i+1 for i in range(N)]

# sigmas
sigmas = [0,1,5,10,15,20]
i = 1

#output
file_out_dir = '/scratch/mmy272/test/scripts/'
i = 1

for n in n_rounds:
    for sigma in sigmas:
         f = open(file_out_dir + '{}.txt'.format(i),'w')
         f.write('/scratch/mmy272/test/data/round_{}/xgb_train_{}.csv,'.format(n,sigma))
         f.write('/scratch/mmy272/test/data/round_{}/xgb_test_{}.csv,'.format(n,sigma))
         
         # creat folder if not exist
         output_dir = '/scratch/mmy272/output/round_{}/sigma_{}/'.format(n,sigma)
         
         
         if not os.path.exists(output_dir):
             os.mkdir(output_dir)
         
         f.write(output_dir)
         f.close()
         i+=1

