#!/bin/bash

#SBATCH --partition=long
#SBATCH -J cpu_fe_mp
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --output=slurm_out/test-%j.out
#SBATCH --error=slurm_out/test-%j.err
ulimit -u

srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
srun -N1 -n1 /cta/users/akindiroglu/workspace/libs/torch_env/bin/python main.py --base_path /scratch/users/akindiroglu/CSL/frames/ --save_path /scratch/users/akindiroglu/CSL/skeleton_mediapipe  --number_of_cores 1 &
wait