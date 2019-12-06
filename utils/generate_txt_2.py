#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:28:32 2019

@author: Jiahao
"""


# number of rounds

# sigmas
sigmas = [1,5,10]

#output
file_out_dir = '/scratch/mmy272/test/scripts2/'
#file_out_dir = '/Users/Jiahao/Desktop/scripts2/'
i = 1


for sigma in sigmas:
     f = open(file_out_dir + '{}.txt'.format(i),'w')
     f.write('/scratch/mmy272/test/data/round_1/xgb_train_{}.csv,'.format(sigma))
     f.write('/scratch/mmy272/test/data/round_1/xgb_test_{}.csv,'.format(sigma))
     
     # creat folder if not exist
     f.write(str(sigma))
     f.close()
     i+=1