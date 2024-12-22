#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=sduss_diff_pp_test
#SBATCH --error=./outputs/sd15-%6j.err
#SBATCH --output=./outputs/sd15-%6j.out
#SBATCH --gres=gpu:L40:1
#SBATCH --partition=Compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hepnode2
#SBATCH --exclusive

/home/zzp/miniconda3/envs/sduss/bin/python ./local_tests/sd15pp_test.py