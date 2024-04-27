#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=sduss_diff_pp_test
#SBATCH --error=./outputs/pp-%6j.err
#SBATCH --output=./outputs/pp-%6j.out
#SBATCH --gres=gpu:L40:1
#SBATCH --partition=Compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hepnode2
#SBATCH --exclusive

/home/zzp/miniconda3/envs/sduss/bin/python ./tests/entrypoints/diff_pp_test_mixed_steps.py 