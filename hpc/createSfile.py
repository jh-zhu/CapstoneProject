#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:38:23 2019

@author: mingmingyu
"""
import numpy as np

n_nodes=10 #number of node
n_tasks=10 #number of tasks
tpn=int(n_tasks/n_nodes) #task per node
cpt=1 #cpu per task
file=open('run-py.s','w')
file.write(f'#!/bin/bash \n\
#SBATCH -N {n_nodes} \n\
#SBATCH -n {n_tasks} \n\
#SBATCH --ntasks-per-node={tpn} \n\
#SBATCH --cpus-per-task={cpt} \n\
#SBATCH --time=3:00:00 \n\
#SBATCH --mem=4GB \n\
#SBATCH --job-name=runPython \n\
#SBATCH --mail-type=END \n\
#SBATCH --output=slurm_%j.out \n\n\
module purge \n\
module load python3/intel/3.7.3 \n\n\
cd /scratch/mmy272/CapstoneProject/core \n\
')

## run python tasks in cluster in parallel
#change the middle part

train_data_path = '/Users/Jiahao/Downloads/train.csv'
test_data_path = '/Users/Jiahao/Downloads/test.csv'
output_directory = '/Users/Jiahao/Downloads/output/'

gammas = np.arange(0.01,0.05,0.01)
C = np.arange(0.1,0.5,0.1)

for gamma,c in zip(gammas,C):
    file.write(f'\
    srun -N 1 -n 1 python3 select_expert.py '+'SVR linear,{},{} {} {} {}'.format(gamma,c,train_data_path,test_data_path,output_directory) +' & \n')

file.write('wait ')

file.close()