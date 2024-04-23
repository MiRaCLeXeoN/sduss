#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=sduss_pp_test
#SBATCH --error=./outputs/%6j.err
#SBATCH --output=./outputs/%6j.out
#SBATCH --gres=gpu:L40:1
#SBATCH --partition=Compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hepnode3
#SBATCH --exclusive

/home/zzp/miniconda3/envs/sduss/bin/python ./tests/entrypoints/diffusion_pipeline_test.py 