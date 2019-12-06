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
file=open('/scratch/mmy272/test/main_script/run.s','w')
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
nums = [4, 4, 6]
gammas, Cs, epsilons = gen_params(nums, "SVR")

for kernel in kernels:
    for gamma in gammas:
        for C in Cs:
            for epsilon in epsilons:
                file.write('srun -N 1 -n 1 python3 select_expert.py SVR {},{},{},{} '.format(kernel, gamma,C, epsilon) +'$SLURM_ARRAY_TASK_ID.txt & \n')

nums = [5, 7]
alphas,l1s = gen_params(nums, "LR")

'''GRID SEARCH'''
for alpha in alphas:
    for l1 in l1s:        
        file.write('srun -N 1 -n 1 python3 select_expert.py LR {},{} '.format(alpha,l1) +'$SLURM_ARRAY_TASK_ID.txt & \n')

nums = [4, 5, 2, 2]
n_estimators, max_depth, min_samples_split, min_samples_leaf = gen_params(nums, "RF")
max_features = ['auto', 'sqrt']

for n in n_estimators:
    for depth in max_depth:
        for split in min_samples_split:
            for leaf in min_samples_leaf:
                for feature in max_features:
                    file.write('srun -N 1 -n 1 python3 select_expert.py RF {},{},{},{},{} '.format(n, depth, split, leaf, feature) +'$SLURM_ARRAY_TASK_ID.txt & \n')            
    
nums = [2, 3, 2, 2, 2, 2, 1, 1]
n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gammas, alphas, lambd = gen_params(nums, "XGBoost")
  
for n in n_estimators:
    for depth in max_depth:
        for l in learning_rate:
            for sample in subsample:
                for bytree in colsample_bytree:
                    for gamma in gammas:
                        for alpha in alphas:
                            for lamb in lambd:
                                file.write('srun -N 1 -n 1 python3 select_expert.py XGBoost {},{},{},{},{},{},{},{} '.format(depth, l,n,sample,bytree, gamma,alpha,lamb) +'$SLURM_ARRAY_TASK_ID.txt & \n')            

           
file.write('wait ')

file.close()