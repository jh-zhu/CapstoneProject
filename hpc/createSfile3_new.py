#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:34:12 2019

@author: mingmingyu
"""

from hpc.generateParameter import *
import numpy as np
from hpc.generateParameter import generateParameter as GP
from core.fileManager import fileName
import random,itertools
import os

def create_random(n_file):
    sFile_pre='/scratch/mmy272/test/main_script3/run'
    sfile=lambda i: sFile_pre+str(i)+'.s'
    
    n_nodes=1 #number of node
    tpn=1 #task per node
    s_script = f'#!/bin/bash \n#SBATCH --nodes={n_nodes} \n#SBATCH --ntasks-per-node={tpn} \n#SBATCH --cpus-per-task=1 \n#SBATCH --time=60:00:00 \n#SBATCH --mem=16GB \n#SBATCH --job-name=runPython \n#SBATCH --error=expert_%A_%a.err \n\nmodule load python3/intel/3.7.3 \n\ncd /scratch/mmy272/test/scripts3 \n'
    file=open(sfile(n_file),'w')
    file.write(s_script)
    
    ## run python tasks in cluster in parallel
    #change the middle part
    
    kernels = ['rbf', 'linear']
    nums={800:[6,6,8], 600:[5,6,6], 400:[5,5,5], 200:[3,3,4], 100:[2,2,4]}
    
    for exp_num,value in nums.items():
        gammas, Cs, epsilons = gen_params_random(value, "SVR")
        num_experts = len(kernels)*len(gammas)*len(Cs)*len(epsilons)
        random_params = random.sample(list(itertools.product(kernels, gammas, Cs, epsilons)), num_experts)
        for params in random_params:
            kernel, gamma,C, epsilon = params
            file.write('srun -N 1 -n 1 python3 select_expert_3.py SVR {},{},{},{} {} '.format(kernel, gamma,C, epsilon,n_file)+str(exp_num) +' $SLURM_ARRAY_TASK_ID.txt & \n')
            
    
    nums={800:[5,7], 600:[5,7], 400:[4,6], 200:[4,6], 100:[3,5]}        
                
    for exp_num,value in nums.items():
        alphas,l1s = gen_params_random(value, "LR")
        num_experts = len(alphas)*len(l1s)
        random_params = random.sample(list(itertools.product(alphas,l1s)), num_experts)
        for params in random_params:
            alpha,l1 = params
            file.write('srun -N 1 -n 1 python3 select_expert_3.py LR {},{} {} '.format(alpha,l1,n_file) +str(exp_num)+' $SLURM_ARRAY_TASK_ID.txt & \n')
            
    
    nums={800:[4,5,2,2], 600:[4,5,2,2], 400:[4,5,2,2], 200:[3,4,2,2], 100:[2,3,2,2]}
                       
    for exp_num,value in nums.items():
        n_estimators, max_depth, min_samples_split, min_samples_leaf = gen_params_random(value, "RF")
        max_features = ['auto', 'sqrt']
        num_experts = len(max_features)*len(n_estimators)*len(max_depth)*len(min_samples_split)*len(min_samples_leaf)
        random_params = random.sample(list(itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)), num_experts)
        for params in random_params:
            n, depth, split, leaf, feature = params
            file.write('srun -N 1 -n 1 python3 select_expert_3.py RF '+ '{},{},{},{},{} {} '.format(n, depth, split, leaf, feature,n_file)+str(exp_num)+' $SLURM_ARRAY_TASK_ID.txt & \n')
            
    file.write('wait ')
    
    file.close()

for n_file in range(100):    
    create_random(n_file)