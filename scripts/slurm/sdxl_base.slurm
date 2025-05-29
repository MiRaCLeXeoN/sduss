#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=sduss_diff_pp_test
#SBATCH --error=./outputs/sdxl-%6j.err
#SBATCH --output=./outputs/sdxl-%6j.out
#SBATCH --gres=gpu:L40:1
#SBATCH --partition=Compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hepnode2
#SBATCH --exclusive

python ./local_tests/sdxlpp_test.py