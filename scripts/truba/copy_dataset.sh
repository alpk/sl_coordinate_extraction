#!/bin/bash

#SBATCH --partition=interactive
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# #SBATCH --mem-per-cpu=10G
#SBATCH -J copy_files
#SBATCH --output=slurm_out/test-%j.out
#SBATCH --error=slurm_out/test-%j.err

rsync -av --delete tam_cluster:/cta/dark/bsign22k/frame-full/ /truba_scratch/akindiroglu/data/bsign22k/frames/
#srun -N1 -n1 -t 01-00:00:00 --partition=interactive   rsync -av --delete tam_cluster:/cta/dark/bsign22k/frame-full/ /truba_scratch/akindiroglu/data/bsign22k/frames/
