#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
# #SBATCH --mem=64000M
#SBATCH -J mediapipe
#SBATCH --output=slurm_out/test-%j.out
#SBATCH --error=slurm_out/test-%j.err


/home/akindiroglu/workspace/libs/miniconda3/envs/pytorch_stable/bin/python main.py --base_path /home/akindiroglu/workspace/data/csl/slr500_rgb/ --save_path /home/akindiroglu/workspace/data/csl/skeleton_mediapipe  --number_of_cores 1

