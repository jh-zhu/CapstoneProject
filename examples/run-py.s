#!/bin/bash 
#SBATCH -N 50 
#SBATCH -n 50 
#SBATCH --ntasks-per-node=5 
#SBATCH --cpus-per-task=1 
#SBATCH --time=1:00:00 
#SBATCH --mem=4GB 
#SBATCH --job-name=runPython 
#SBATCH --mail-type=END 
#SBATCH --output=slurm_%j.out 

module purge 
module load python3/intel/3.7.3 

cd /scratch/mmy272/test/CapstoneProject/core 
