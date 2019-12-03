#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:38:23 2019

@author: mingmingyu
"""
import numpy as np
from hpc.generateParameter import generateParameter as GP
from core.fileManager import fileName
import os

n_nodes=50 #number of node
n_tasks=50 #number of tasks
tpn=int(n_tasks/n_nodes) #task per node
cpt=1 #cpu per task
file=open('run-py.s','w')
file.write(f'#!/bin/bash \n\
#SBATCH -N {n_nodes} \n\
#SBATCH -n {n_tasks} \n\
#SBATCH --ntasks-per-node={tpn} \n\
#SBATCH --cpus-per-task={cpt} \n\
#SBATCH --time=1:00:00 \n\
#SBATCH --mem=4GB \n\
#SBATCH --job-name=runPython \n\
#SBATCH --mail-type=END \n\
#SBATCH --output=slurm_%j.out \n\n\
module purge \n\
module load python3/intel/3.7.3 \n\n\
cd /scratch/mmy272/test/CapstoneProject/core \n\
')

## run python tasks in cluster in parallel
#change the middle part
filename = fileName()
data_dir = filename.data_folder
output_dir = filename.output_folder

train_data_path = data_dir+'xgb_train_'
test_data_path = data_dir+'xgb_test_'


sigmas = [0,5,20]

modelName='SVR'
kernel = 'rbf'
gammas = GP(-15,3,9).exp()
C = GP(-5,15,10).exp()

for gamma in gammas:
    for c in C:
        for sigma in sigmas:
            read_train=train_data_path+str(sigma)+'.csv'
            read_test=test_data_path+str(sigma)+'.csv'
            output=output_dir+str(sigma)+'/'
            
            if not os.path.exists(output):
                os.makedirs(output)
            
            file.write(f'\
    srun -N 1 -n 1 python3 select_expert.py '+modelName+' '+kernel+',{},{} {} {} {}'.format(gamma,c,read_train,read_test,output) +' & \n')


file.write('wait ')

file.close()