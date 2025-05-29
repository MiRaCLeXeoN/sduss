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

export TORCH_INCLUDE_PATH="/home/zzp/miniconda3/envs/sduss/lib/python3.9/site-packages/torch/include"

/home/zzp/miniconda3/envs/sduss/bin/python ./sduss/entrypoints/api_server.py \
    --model_name_or_pth "/data/home/zzp/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9" \
    --policy "fcfs_mixed" \
    --use_esymred \
    --use_mixed_precision \
    --max_batchsize 16 \
    --non_blocking_step \
    --overlap_prepare \
    --torch_dtype "float16" \
    --engine_use_ray