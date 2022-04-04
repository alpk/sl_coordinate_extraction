#!/bin/bash

#SBATCH --partition=mid2
#SBATCH --time=08-00:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
# #SBATCH --mem-per-cpu=10G
#SBATCH -J cpu_fe_mp
#SBATCH --output=slurm_out/test-%j.out
#SBATCH --error=slurm_out/test-%j.err

srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &
srun -N1 -n1 /truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py --base_path /truba_scratch/akindiroglu/data/bsign22k/frames/ --save_path /truba_scratch/akindiroglu/data/bsign22k/skeleton_mediapipe/ --number_of_cores 1  &


wait

