#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=sduss_diff_pp_test
#SBATCH --error=./outputs/sduss-%6j.err
#SBATCH --output=./outputs/sduss-%6j.out
#SBATCH --gres=gpu:L40:1
#SBATCH --partition=Compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hepnode3

export SDUSS_COLLECT_DATA=true
export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

export TORCH_INCLUDE_PATH="/home/zzp/miniconda3/envs/sduss/lib/python3.9/site-packages/torch/include"

if [ ${MODEL} == "sd1.5" ]; then
    MODEL_PATH="/data/home/zzp/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"
elif [ ${MODEL} == "sdxl" ]; then
    MODEL_PATH="/data/home/zzp/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
fi

/home/zzp/miniconda3/envs/sduss/bin/python ./sduss/entrypoints/api_server.py \
    --model_name_or_pth ${MODEL_PATH} \
    --policy ${POLICY} \
    ${USE_MIXED_PRECISION} \
    ${OVERLAP_PREPARE} \
    ${NON_BLOCKING_STEP} \
    --use_esymred \
    --max_batchsize 16 \
    --torch_dtype "float16" \
    --engine_use_ray