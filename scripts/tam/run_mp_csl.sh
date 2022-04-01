#!/bin/bash

#SBATCH --partition=longer
#SBATCH -N 1
#SBATCH -n 10
# #SBATCH --mem=64000M
#SBATCH -J cpu_fe_mp
#SBATCH --output=slurm_out/test-%j.out
#SBATCH --error=slurm_out/test-%j.err


/cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 10

