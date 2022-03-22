#!/bin/bash

#SBATCH --partition=long
#SBATCH -N 1
#SBATCH -n 10
# #SBATCH --mem=64000M
#SBATCH -J mediapipe
#SBATCH --output=slurm_out/test-%j.out
#SBATCH --error=slurm_out/test-%j.err


/cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /cta/dark/CSLR/frames/ --save_path /cta/dark/CSLR/skeleton_mediapipe  --number_of_cores 10

