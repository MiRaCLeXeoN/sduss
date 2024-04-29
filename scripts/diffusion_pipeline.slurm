#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=sduss_diff_pp_test
#SBATCH --error=./outputs/pp-%6j.err
#SBATCH --output=./outputs/pp-%6j.out
#SBATCH --gres=gpu:L40:1
#SBATCH --partition=Compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hepnode3
#SBATCH --exclusive

export TORCH_INCLUDE_PATH="/home/zzp/miniconda3/envs/sduss/lib/python3.9/site-packages/torch/include"

/home/zzp/miniconda3/envs/sduss/bin/python ./tests/pipelines/esymred_sd15_test2.py 