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
tpn=5 #task per node
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

sigmas = [0,5,20,100]
points_grid=10000

gammas = GP(-15,3,10).exp()
Cs = GP(-5,15,10).exp()

for gamma in gammas:
    for C in Cs:
        for sigma in sigmas:
            read_train=train_data_path+str(sigma)+'.csv'
            read_test=test_data_path+str(sigma)+'.csv'
            output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
            
            if not os.path.exists(output):
                os.makedirs(output)
            
            file.write(f'srun -N 1 -n 1 python3 select_expert.py SVR '+ 'rbf,{},{} {} {} {}'.format(gamma,C,read_train,read_test,output) +' & \n')

alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
l1_ratio=np.arange(0.0, 1.0, 0.1)

for alpha in alphas:
    for l1 in l1_ratio:
        for sigma in sigmas:
            read_train=train_data_path+str(sigma)+'.csv'
            read_test=test_data_path+str(sigma)+'.csv'
            output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
            
            file.write(f'\srun -N 1 -n 1 python3 select_expert.py LR '+'{},{} {} {} {}'.format(alpha,l1,read_train,read_test,output) +' & \n')

n_estimators=[int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_depth=[int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split=[2, 5, 10]
min_samples_leaf=[1,2,4]
max_features=['auto', 'sqrt']

for n in n_estimators:
    for depth in max_depth:
        for split in min_samples_split:
            for leaf in min_samples_leaf:
                for feature in max_features:
                    for sigma in sigmas:
                        read_train=train_data_path+str(sigma)+'.csv'
                        read_test=test_data_path+str(sigma)+'.csv'
                        output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
                                                
                        file.write(f'srun -N 1 -n 1 python3 select_expert.py RF '+ '{},{},{},{},{} {} {} {}'.format(n, depth, split, leaf, feature,read_train,read_test,output) +' & \n')            

max_depth2=[int(x) for x in np.linspace(start = 3, stop = 7, num = 4)]
learning_rate=GP(0.01,0.2,4).grid()
n_estimators2=[100,250,500,1000]
subsample=GP(0.5,1,4).grid()
colsample_bytree=GP(0.5,1,4).grid()
gamma2=[0.01,1]
alpha2=[0.0,1]
lambd=[0.0,5]
  
for n in n_estimators2:
    for depth in max_depth2:
        for l in learning_rate:
            for sample in subsample:
                for bytree in colsample_bytree:
                    for gamma in gamma2:
                        for alpha in alpha2:
                            for lamb in lambd:
                                for sigma in sigmas:
                                    read_train=train_data_path+str(sigma)+'.csv'
                                    read_test=test_data_path+str(sigma)+'.csv'
                                    output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
                                                            
                                    file.write(f'srun -N 1 -n 1 python3 select_expert.py XGBoost '+ '{},{},{},{},{},{},{},{} {} {} {}'.format(depth, l,n,sample,bytree, gamma,alpha,lamb,read_train,read_test,output) +' & \n')            

               
file.write('wait ')

file.close()