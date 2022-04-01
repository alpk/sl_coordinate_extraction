#!/bin/bash

#SBATCH --partition=mid2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
# #SBATCH --mem-per-cpu=10G
#SBATCH -J cpu_fe_mp
#SBATCH --output=slurm_out/test-%j.out
#SBATCH --error=slurm_out/test-%j.err

srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py \
            --base_path /truba_scratch/akindiroglu/data/CSL/frames/ \
            --save_path /truba_scratch/akindiroglu/data/CSL/skeleton_mediapipe/ \
            --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py \
            --base_path /truba_scratch/akindiroglu/data/CSL/frames/ \
            --save_path /truba_scratch/akindiroglu/data/CSL//skeleton_mediapipe/ \
            --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py \
            --base_path /truba_scratch/akindiroglu/data/CSL/frames/ \
            --save_path /truba_scratch/akindiroglu/data/CSL//skeleton_mediapipe/ \
            --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py \
            --base_path /truba_scratch/akindiroglu/data/CSL/frames/ \
            --save_path /truba_scratch/akindiroglu/data/CSL//skeleton_mediapipe/ \
            --number_of_cores 1  &

wait

