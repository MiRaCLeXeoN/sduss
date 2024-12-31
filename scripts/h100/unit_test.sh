
# ulimit -n 4096

export SDUSS_COLLECT_DATA=true
export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"
export ESYMRED_UPSAMPLE_PATH="./exp/$MODEL-upsample-threshold0.01.pkl"
export ESYMRED_DOWNSAMPLE_PATH="./exp/$MODEL-downsample-threshold0.01.pkl"

export TORCH_INCLUDE_PATH="/workspace/local_conda_env/opt/conda/envs/sduss/lib/python3.9/site-packages/torch/include"

if [ ${MODEL} == "sd1.5" ]; then
    export MODEL_PATH="/workspace/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1"
elif [ ${MODEL} == "sdxl" ]; then
    export MODEL_PATH="/workspace/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
fi

python ./sduss/entrypoints/api_server.py \
    --model_name_or_pth ${MODEL_PATH} \
    --policy ${POLICY} \
    ${USE_MIXED_PRECISION} \
    ${OVERLAP_PREPARE} \
    ${NON_BLOCKING_STEP} \
    --use_esymred \
    --max_batchsize 12 \
    --torch_dtype "float16" \
    --engine_use_mp \
    --worker_use_mp