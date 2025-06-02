export MODEL="sdxl"
export NUM=500

if [[ $MODEL == "sd3" ]]; then
    export SLO="5"
elif [[ $MODEL == "sdxl" ]]; then
    export SLO="5"
fi

qps=1.2
dp_size=1
export GPUS="[0-7]"
export DATA_PARALLEL_SIZE=${dp_size}
export QPS=$(awk "BEGIN {printf \"%.1f\", $qps * $dp_size}")

#export POLICY="orca_resbyres"
# export POLICY="fcfs_mixed"
export POLICY="esymred"

if [[ $POLICY == "esymred" || $POLICY == "fcfs_mixed" ]]; then
    export USE_MIXED_PRECISION="--use_mixed_precision"
    export ESYMRED_USE_CACHE="TRUE"
    # export ESYMRED_USE_CACHE="FALSE"
elif [[ $POLICY == "orca_resbyres" ]]; then
    export USE_MIXED_PRECISION=""
    export ESYMRED_USE_CACHE="FALSE"
else
    export USE_MIXED_PRECISION=""
fi
result_dir_path="./results/${MODEL}/${QPS}_${SLO}_${POLICY}_${DATA_PARALLEL_SIZE}"
if [ -d "${result_dir_path}" ]; then
    find ${result_dir_path} -type f -delete
else
    mkdir -p "${result_dir_path}"
fi
# echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION, SLO=$SLO"

export SDUSS_COLLECT_DATA=true
export ESYMRED_PREDICTOR_PATH="./exp/schedule_predictor_$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

export ESYMRED_UPSAMPLE_PATH="./exp/$MODEL-upsample-threshold0.01.pkl"
export ESYMRED_DOWNSAMPLE_PATH="./exp/$MODEL-downsample-threshold0.01.pkl"
export ESYMRED_TRANSFORMER_PATH="./exp/$MODEL-state-threshold0.01.pkl"

export TORCH_INCLUDE_PATH="/opt/conda/envs/sduss/lib/python3.9/site-packages/torch/include"

if [ ${MODEL} == "sd1.5" ]; then
    export MODEL_PATH="/workspace/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1"
elif [ ${MODEL} == "sdxl" ]; then
    export MODEL_PATH="/workspace/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
elif [ ${MODEL} == "sd3" ]; then
    export MODEL_PATH="/workspace/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
fi

echo "Start ${result_dir_path}"

python ./tests/server/direct_test.py \
    --model_name_or_pth ${MODEL_PATH} \
    --policy ${POLICY} \
    --dispatcher_policy greedy \
    ${USE_MIXED_PRECISION} \
    --use_esymred \
    --max_batchsize 12 \
    --torch_dtype "float16" \
    --engine_use_mp \
    --worker_use_mp \
    --data_parallel_size ${DATA_PARALLEL_SIZE} \
    --gpus "${GPUS}" \
    --model ${MODEL} \
    --qps ${QPS} \
    --SLO ${SLO} \
    --host localhost \
    --port 8000 \
    --num ${NUM} > unit_test.log 2>&1

ps aux | grep sduss | grep -v grep | awk '{print $2}' | xargs kill -9
# echo "cancelled job $job_num"
# cp -r "${result_dir_path}" /workspace/results/${MODEL}/