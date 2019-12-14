#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:50:28 2019

@author: yitongcai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:38:23 2019

@author: mingmingyu
"""
from hpc.generateParameter import gen_params, gen_params_random
#generateParameter as GP
from core.fileManager import fileName
import os
import random,itertools  

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

sigmas = [5]
points_grid=10000
 
'''*********************************************************************************
    Support Vector Machine
   *********************************************************************************''' 
# number: 200
kernels = ['rbf', 'linear']
#gammas = GP(-4,3,4).grid_log()
#Cs = GP(-3,4,4).grid_log()
#epsilons = GP(-6,2,6).grid_log(base=2)

'''GRID SEARCH'''
nums = [4, 4, 6]
gammas, Cs, epsilons = gen_params(nums, "SVR")

for kernel in kernels:
    for gamma in gammas:
        for C in Cs:
            for epsilon in epsilons:
                for sigma in sigmas:
                    read_train=train_data_path+str(sigma)+'.csv'
                    read_test=test_data_path+str(sigma)+'.csv'
                    output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
                    
                    if not os.path.exists(output):
                        os.makedirs(output)
                    
                    file.write(f'srun -N 1 -n 1 python3 select_expert.py SVR '+ '{},{},{},{} {} {} {}'.format(kernel, gamma,C, epsilon, read_train,read_test,output) +' & \n')

'''RANDOM SEARCH'''
nums = [4, 4, 6]
gammas, Cs, epsilons = gen_params_random(nums, "SVR")
 
num_experts = len(kernels)*len(gammas)*len(Cs)*len(epsilons)
random_params = random.sample(set(itertools.product(kernels, gammas, Cs, epsilons)), num_experts)

for params in random_params:
    kernel, gamma,C, epsilon = params
    for sigma in sigmas:
        read_train=train_data_path+str(sigma)+'.csv'
        read_test=test_data_path+str(sigma)+'.csv'
        output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
        
        if not os.path.exists(output):
            os.makedirs(output)
        
        file.write(f'srun -N 1 -n 1 python3 select_expert.py SVR '+ '{},{},{},{} {} {} {}'.format(kernel, gamma,C, epsilon, read_train,read_test,output) +' & \n')



'''*********************************************************************************
    Linear Regression
   *********************************************************************************''' 
# number: 35
#alphas=GP(-4,2,5).grid_log()
#l1s=GP(0,1,7).grid_lin()

'''GRID SEARCH'''
nums = [5, 7]
alphas,l1s = gen_params(nums, "LR")

for alpha in alphas:
    for l1 in l1s:
        for sigma in sigmas:
            read_train=train_data_path+str(sigma)+'.csv'
            read_test=test_data_path+str(sigma)+'.csv'
            output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
            
            file.write(f'srun -N 1 -n 1 python3 select_expert.py LR '+'{},{} {} {} {}'.format(alpha,l1,read_train,read_test,output) +' & \n')

'''RANDOM SEARCH'''
nums = [5, 7]
alphas,l1s = gen_params_random(nums, "LR")

num_experts = len(alphas)*len(l1s)
random_params = random.sample(set(itertools.product(alphas,l1s)), num_experts)

for params in random_params:
    alpha,l1 = params
    for sigma in sigmas:
        read_train=train_data_path+str(sigma)+'.csv'
        read_test=test_data_path+str(sigma)+'.csv'
        output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
        
        file.write(f'srun -N 1 -n 1 python3 select_expert.py LR '+'{},{} {} {} {}'.format(alpha,l1,read_train,read_test,output) +' & \n')



'''*********************************************************************************
    Random Forest
   *********************************************************************************'''
# number: 80
#n_estimators = GP(100,1000,4).grid_lin("int")
#max_depth = GP(2,15,5).grid_lin("int")
#min_samples_split = GP(2,15,2).grid_lin("int")
#min_samples_leaf = GP(1,10,2).grid_lin("int")

'''GRID SEARCH'''  
nums = [4, 5, 2, 2]
n_estimators, max_depth, min_samples_split, min_samples_leaf = gen_params(nums, "RF")
max_features = ['auto', 'sqrt']

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

'''RANDOM SEARCH'''
nums = [4, 5, 2, 2]
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
                                
        file.write(f'srun -N 1 -n 1 python3 select_expert.py RF '+ '{},{},{},{},{} {} {} {}'.format(n, depth, split, leaf, feature,read_train,read_test,output) +' & \n')            



'''*********************************************************************************
    XGBoost
   *********************************************************************************'''     
# number: 96
   
'''GRID SEARCH'''
nums = [2, 3, 2, 2, 2, 2, 1, 1]

n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, alpha, lambd = gen_params(nums, "XGBoost")
  
for n in n_estimators:
    for depth in max_depth:
        for l in learning_rate:
            for sample in subsample:
                for bytree in colsample_bytree:
                    for gamma in gamma:
                        for alpha in alpha:
                            for lamb in lambd:
                                for sigma in sigmas:
                                    read_train=train_data_path+str(sigma)+'.csv'
                                    read_test=test_data_path+str(sigma)+'.csv'
                                    output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
                                                            
                                    file.write(f'srun -N 1 -n 1 python3 select_expert.py XGBoost '+ '{},{},{},{},{},{},{},{} {} {} {}'.format(depth, l,n,sample,bytree, gamma,alpha,lamb,read_train,read_test,output) +' & \n')            


'''RANDOM SEARCH'''
nums = [2, 3, 2, 2, 2, 2, 1, 1]
n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, alpha, lambd = gen_params_random(nums, "XGBoost")
num_experts = len(n_estimators)*len(max_depth)*len(learning_rate)*len(subsample)*len(colsample_bytree)*len(gamma)*len(alpha)*len(lambd)
random_params = random.sample(set(itertools.product(n_estimators, max_depth, learning_rate, subsample,
                                                    colsample_bytree, gamma, alpha, lambd)), num_experts)

for params in random_params:
    depth, l,n,sample,bytree, gamma,alpha,lamb = params
    for sigma in sigmas:
        read_train=train_data_path+str(sigma)+'.csv'
        read_test=test_data_path+str(sigma)+'.csv'
        output=output_dir+str(points_grid)+'/'+str(sigma)+'/'
                                
        file.write(f'srun -N 1 -n 1 python3 select_expert.py XGBoost '+ '{},{},{},{},{},{},{},{} {} {} {}'.format(depth, l,n,sample,bytree, gamma,alpha,lamb,read_train,read_test,output) +' & \n')            



               
file.write('wait ')

file.close()