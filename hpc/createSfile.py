#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:38:23 2019

@author: mingmingyu
"""
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
file.write(f'\
srun -N 1 -n 1 python3 select_expert.py '+'SVR rbf,0.1,1' +' & \n\
')

file.write('wait ')

file.close()