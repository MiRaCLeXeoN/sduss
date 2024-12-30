
# ulimit -n 4096

export SDUSS_COLLECT_DATA=true
export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

export TORCH_INCLUDE_PATH="/root/miniconda3/envs/sduss/lib/python3.9/site-packages/torch/include"

if [ ${MODEL} == "sd1.5" ]; then
    export MODEL_PATH="/workspace/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1"
elif [ ${MODEL} == "sdxl" ]; then
    export MODEL_PATH="/workspace/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
fi

python ./sduss/entrypoints/api_server.py \
    --model_name_or_pth ${MODEL_PATH} \
    --policy ${POLICY} \
    --dispatcher_policy greedy \
    ${USE_MIXED_PRECISION} \
    --use_esymred \
    --max_batchsize 32 \
    --torch_dtype "float16" \
    --engine_use_mp \
    --worker_use_mp \
    --data_parallel_size ${DATA_PARALLEL_SIZE} \
    --gpus "${GPUS}"