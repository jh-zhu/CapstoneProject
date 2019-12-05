#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:30:24 2019

@author: mingmingyu
"""

from core.fileManager import fileName
filename = fileName()
data_dir = filename.data_folder
output_dir = filename.output_folder
scripts=filename.scripts

rounds=[str(i)+'/' for i in range(1,101)]
sigmas = [0,1,5,10,15,20]

job_id = 1

kernels = ['rbf', 'linear']

nums = [4, 4, 6]
gammas, Cs, epsilons = gen_params(nums, "SVR")

'''GRID SEARCH'''
for rnd in rounds:
    for kernel in kernels:
        for gamma in gammas:
            for C in Cs:
                for epsilon in epsilons:
                    for sigma in sigmas:
                        read_train=data_dir+'round_'+rnd+'xgb_train_'+str(sigma)+'.csv'
                        read_test=data_dir+'round_'+rnd+'xgb_test_'+str(sigma)+'.csv'
                        output=output_dir+'round_'+rnd+str(sigma)+'/'
                        
                        
                        if not os.path.exists(output):
                            os.makedirs(output)
                        
                        File_object = open(scripts + str(job_id)+'.txt','w')
                        File_object.write(f'SVR '+ '{},{},{},{} {} {} {}'.format(kernel, gamma,C, epsilon, read_train,read_test,output) )
                        File_object.close()
                        job_id+=1
# =============================================================================
# 
# File_object = open(root + str(job_id)+'.txt','w')
# File_object.write('XGBoost 3,3,1,1,1,1,3,3 /scratch/mmy272/test/CapstoneProject/data/xgb_train_20.csv /scratch/mmy272/test/CapstoneProject/data/xgb_test_20.csv /scratch/mmy272/test/output/10000/20/')
# File_object.close()
# job_id = 2
# File_object = open(root + str(job_id)+'.txt','w')
# File_object.write('RF 400,10,2,1,auto /scratch/mmy272/test/CapstoneProject/data/xgb_train_20.csv /scratch/mmy272/test/CapstoneProject/data/xgb_test_20.csv /scratch/mmy272/test/output/10000/20/')
# File_object.close()
# job_id = 3
# File_object = open(root + str(job_id)+'.txt','w')
# File_object.write('LR 10,0.0 /scratch/mmy272/test/CapstoneProject/data/xgb_train_20.csv /scratch/mmy272/test/CapstoneProject/data/xgb_test_20.csv /scratch/mmy272/test/output/10000/20/')
# File_object.close()
# 'SVR rbf,1,2 /scratch/mmy272/test/data/round_1/xgb_train_0.csv /scratch/mmy272/test/data/roudn_1/xgb_test_0.csv /scratch/mmy272/test/output/round_?/sigma_?/G?R/?#xp/
# '
# =============================================================================
