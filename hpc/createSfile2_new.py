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

sFile_pre='/scratch/mmy272/test/main_script2/run'
sfile=lambda i: sFile_pre+str(i)+'.s'

n_file = 1 
n_nodes=1 #number of node
tpn=1 #task per node
s_script = f'#!/bin/bash \n\
#SBATCH --nodes={n_nodes} \n\
#SBATCH --ntasks-per-node={tpn} \n\
#SBATCH --cpus-per-task=1 \n\
#SBATCH --time=60:00:00 \n\
#SBATCH --mem=16GB \n\
#SBATCH --job-name=runPython \n\
#SBATCH --error=expert_%A_%a.err \n\n\
module load python3/intel/3.7.3 \n\n\
cd /scratch/mmy272/test/scripts2 \n\
'
file=open(sfile(n_file),'w')
file.write(s_script)

## run python tasks in cluster in parallel
#change the middle part

n_line=0

kernels = ['rbf', 'linear']
nums={800:[6,6,8], 600:[5,6,6], 400:[5,5,5], 200:[3,3,4], 100:[2,2,4]}

for exp_num,value in nums.items():
    gammas, Cs, epsilons = gen_params(value, "SVR")
    for kernel in kernels:
        for gamma in gammas:
            for C in Cs:
                for epsilon in epsilons:
                    file.write('srun -N 1 -n 1 python3 select_expert_2.py SVR {},{},{},{} G '.format(kernel, gamma,C, epsilon) +str(exp_num)+' $SLURM_ARRAY_TASK_ID.txt & \n')
                    n_line+=1
                    if n_line%500==0:
                        file.write('wait ')
                        file.close()
                        n_file+=1
                        file=open(sfile(n_file),'w')
                        file.write(s_script)

for exp_num,value in nums.items():
    gammas, Cs, epsilons = gen_params_random(value, "SVR")
    num_experts = len(kernels)*len(gammas)*len(Cs)*len(epsilons)
    random_params = random.sample(list(itertools.product(kernels, gammas, Cs, epsilons)), num_experts)
    for params in random_params:
        kernel, gamma,C, epsilon = params
        file.write('srun -N 1 -n 1 python3 select_expert_2.py SVR {},{},{},{} R '.format(kernel, gamma,C, epsilon)+str(exp_num) +' $SLURM_ARRAY_TASK_ID.txt & \n')
        n_line+=1
        if n_line%500==0:
            file.write('wait ')
            file.close()
            n_file+=1
            file=open(sfile(n_file),'w')
            file.write(s_script)

nums={800:[5,7], 600:[5,7], 400:[4,6], 200:[4,6], 100:[3,5]}
for exp_num,value in nums.items():
    alphas, l1s = gen_params(value, "LR")
    for alpha in alphas:
        for l1 in l1s:
            file.write('srun -N 1 -n 1 python3 select_expert_2.py LR {},{} G '.format(alpha,l1) +str(exp_num)+' $SLURM_ARRAY_TASK_ID.txt & \n')
            n_line+=1
            if n_line%500==0:
                file.write('wait ')
                file.close()
                n_file+=1
                file=open(sfile(n_file),'w')
                file.write(s_script)
            
for exp_num,value in nums.items():
    alphas,l1s = gen_params_random(value, "LR")
    num_experts = len(alphas)*len(l1s)
    random_params = random.sample(list(itertools.product(alphas,l1s)), num_experts)
    for params in random_params:
        alpha,l1 = params
        file.write('srun -N 1 -n 1 python3 select_expert_2.py LR {},{} R '.format(alpha,l1) +str(exp_num)+' $SLURM_ARRAY_TASK_ID.txt & \n')
        n_line+=1
        if n_line%500==0:
            file.write('wait ')
            file.close()
            n_file+=1
            file=open(sfile(n_file),'w')
            file.write(s_script)

nums={800:[4,5,2,2], 600:[4,5,2,2], 400:[4,5,2,2], 200:[3,4,2,2], 100:[2,3,2,2]}

for exp_num,value in nums.items():
    n_estimators, max_depth, min_samples_split, min_samples_leaf = gen_params(value, "RF")
    max_features = ['auto', 'sqrt']
    for n in n_estimators:
        for depth in max_depth:
            for split in min_samples_split:
                for leaf in min_samples_leaf:
                    for feature in max_features:
                        file.write('srun -N 1 -n 1 python3 select_expert_2.py RF {},{},{},{},{} G '.format(n, depth, split, leaf, feature)+str(exp_num)+' $SLURM_ARRAY_TASK_ID.txt & \n')
                        n_line+=1
                        if n_line%500==0:
                            file.write('wait ')
                            file.close()
                            n_file+=1
                            file=open(sfile(n_file),'w')
                            file.write(s_script)

for exp_num,value in nums.items():
    n_estimators, max_depth, min_samples_split, min_samples_leaf = gen_params_random(value, "RF")
    max_features = ['auto', 'sqrt']
    num_experts = len(max_features)*len(n_estimators)*len(max_depth)*len(min_samples_split)*len(min_samples_leaf)
    random_params = random.sample(list(itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)), num_experts)
    for params in random_params:
        n, depth, split, leaf, feature = params
        file.write('srun -N 1 -n 1 python3 select_expert_2.py RF '+ '{},{},{},{},{} R '.format(n, depth, split, leaf, feature)+str(exp_num)+' $SLURM_ARRAY_TASK_ID.txt & \n')
        n_line+=1
        if n_line%500==0:
            file.write('wait ')
            file.close()
            n_file+=1
            file=open(sfile(n_file),'w')
            file.write(s_script)

file.write('wait ')

file.close()
                                             