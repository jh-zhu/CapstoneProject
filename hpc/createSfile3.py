#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:38:23 2019

@author: mingmingyu
"""
from generateParameter import *
import numpy as np
from hpc.generateParameter import generateParameter as GP
from core.fileManager import fileName
import os

n_nodes=5 #number of node
tpn=4 #task per node
file=open('/scratch/mmy272/test/main_script2/run.s','w')
file.write(f'#!/bin/bash \n\
#SBATCH --nodes={n_nodes} \n\
#SBATCH --ntasks-per-node={tpn} \n\
#SBATCH ----cpus-per-task=1 \n\
#SBATCH --time=12:00:00 \n\
#SBATCH --mem=16GB \n\
#SBATCH --job-name=runPython \n\
#SBATCH --error=expert_%A_%a.err \n\n\
module purge \n\
module load python3/intel/3.7.3 \n\n\
cd /scratch/mmy272/test/scripts \n\
')

## run python tasks in cluster in parallel
#change the middle part

kernels = ['rbf', 'linear']
nums={800:[6,6,8], 600:[5,6,6], 400:[5,5,5], 200:[3,3,4], 100:[2,2,4]}
gammas, Cs, epsilons = gen_params(nums, "SVR")

for exp_num,value in nums.items():
    gammas, Cs, epsilons = gen_params(value, "SVR")
    for kernel in kernels:
        for gamma in gammas:
            for C in Cs:
                for epsilon in epsilons:
                    file.write('srun -N 1 -n 1 python3 select_expert.py SVR {},{},{},{} G'.format(kernel, gamma,C, epsilon) +str(exp_num)+'$SLURM_ARRAY_TASK_ID.txt & \n')



for exp_num,value in nums.items():
    gammas, Cs, epsilons = gen_params_random(value, "SVR")
    num_experts = len(kernels)*len(gammas)*len(Cs)*len(epsilons)
    random_params = random.sample(set(itertools.product(kernels, gammas, Cs, epsilons)), num_experts)
    for params in random_params:
        kernel, gamma,C, epsilon = params
        file.write('srun -N 1 -n 1 python3 select_expert.py SVR {},{},{},{} R'.format(kernel, gamma,C, epsilon)+str(exp_num) +'$SLURM_ARRAY_TASK_ID.txt & \n')



nums={800:[5,7], 600:[5,7], 400:[4,6], 200:[4,6], 100:[3,5]}
for exp_num,value in nums.items():
    alphas, l1s = gen_params(nums, "LR")
    for alpha in alphas:
        for l1 in l1s:        
            file.write('srun -N 1 -n 1 python3 select_expert.py LR {},{} G'.format(alpha,l1) +str(exp_num)+'$SLURM_ARRAY_TASK_ID.txt & \n')


for exp_num,value in nums.items():
    alphas,l1s = gen_params_random(nums, "LR")
    num_experts = len(alphas)*len(l1s)
    random_params = random.sample(set(itertools.product(alphas,l1s)), num_experts)
    for params in random_params:
        alpha,l1 = params
        for alpha in alphas:
            for l1 in l1s:        
                file.write('srun -N 1 -n 1 python3 select_expert.py LR {},{} R'.format(alpha,l1) +str(exp_num)+'$SLURM_ARRAY_TASK_ID.txt & \n')



nums={800:[4,5,2,2], 600:[4,5,2,2], 400:[4,5,2,2], 200:[3,4,2,2], 100:[2,3,2,2]}

for exp_num,value in nums.items():
    n_estimators, max_depth, min_samples_split, min_samples_leaf = gen_params(nums, "RF")
    max_features = ['auto', 'sqrt']
    for n in n_estimators:
        for depth in max_depth:
            for split in min_samples_split:
                for leaf in min_samples_leaf:
                    for feature in max_features:
                        file.write('srun -N 1 -n 1 python3 select_expert.py RF {},{},{},{},{} G'.format(n, depth, split, leaf, feature)+str(exp_num)+'$SLURM_ARRAY_TASK_ID.txt & \n')           


for exp_num,value in nums.items():
    n_estimators, max_depth, min_samples_split, min_samples_leaf = gen_params_random(nums, "RF")
    max_features = ['auto', 'sqrt']
    num_experts = len(max_features)*len(n_estimators)*len(max_depth)*len(min_samples_split)*len(min_samples_leaf)
    random_params = random.sample(set(itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)), num_experts)
    for params in random_params:
        n, depth, split, leaf, feature = params
        for sigma in sigmas:
            read_train=train_data_path+str(sigma)+'.csv'
            read_test=test_data_path+str(sigma)+'.csv'
            output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
                                    
            file.write('srun -N 1 -n 1 python3 select_expert.py RF '+ '{},{},{},{},{} {} {} {} R'.format(n, depth, split, leaf, feature,read_train,read_test,output)+str(exp_num)+'$SLURM_ARRAY_TASK_ID.txt & \n')      

        
# =============================================================================
# nums = [2, 3, 2, 2, 2, 2, 1, 1]
# n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gammas, alphas, lambd = gen_params(nums, "XGBoost")
#   
# for n in n_estimators:
#     for depth in max_depth:
#         for l in learning_rate:
#             for sample in subsample:
#                 for bytree in colsample_bytree:
#                     for gamma in gammas:
#                         for alpha in alphas:
#                             for lamb in lambd:
#                                 file.write('srun -N 1 -n 1 python3 select_expert.py XGBoost {},{},{},{},{},{},{},{} '.format(depth, l,n,sample,bytree, gamma,alpha,lamb) +'$SLURM_ARRAY_TASK_ID.txt & \n')            
# 
#            
# =============================================================================

file.write('wait ')

file.close()