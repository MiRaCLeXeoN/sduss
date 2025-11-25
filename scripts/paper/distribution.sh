set -e

# For figure 12

MODEL_LIST=("sdxl" "sd3")
DP_LIST="8"

for model in "${MODEL_LIST[@]}"; do
    if [[ $model == "sd3" ]]; then
        QPS_LIST=(0.4)
        POLICY_LIST=("esymred" "fcfs_mixed" "orca_resbyres")
    elif [[ $model == "sdxl" ]]; then
        QPS_LIST=(1.1)
        POLICY_LIST=("esymred" "fcfs_mixed" "orca_resbyres")
    fi

    export SLO="5"
    export MODEL=$model
    export NUM=$NUM

    distribution_list=("small" "medium" "large")

    for dp_size in $DP_LIST; do
        for policy_name in $POLICY_LIST; do 
            for qps in $QPS_LIST; do
                for distribution in "${distribution_list[@]}"; do
                    export QPS=$(awk "BEGIN {printf \"%.1f\", $qps * $dp_size}")
                    export QPS="${QPS}_${distribution}"
                    export POLICY=$policy_name
                    export DATA_PARALLEL_SIZE=$dp_size
                    # last_gpu=$(awk "BEGIN {printf $dp_size - 1}")
                    # export GPUS="[0-${last_gpu}]"
                    start_gpu=$(awk "BEGIN {printf 8 - $dp_size}")
                    export GPUS="[${start_gpu}-7]"

                    if [[ $POLICY == "esymred" || $POLICY == "fcfs_mixed" ]]; then
                        export USE_MIXED_PRECISION="--use_mixed_precision"
                        export ESYMRED_USE_CACHE="TRUE"
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

                    export SDUSS_COLLECT_DATA=true
                    export ESYMRED_PREDICTOR_PATH="./exp/schedule_predictor_$MODEL.pkl"
                    export ESYMRED_EXEC_TIME_DIR="./exp/profile"

                    export ESYMRED_UPSAMPLE_PATH="./exp/$MODEL-upsample-threshold0.01.pkl"
                    export ESYMRED_DOWNSAMPLE_PATH="./exp/$MODEL-downsample-threshold0.01.pkl"
                    export ESYMRED_TRANSFORMER_PATH="./exp/$MODEL-state-threshold0.01.pkl"

                    if [ ${MODEL} == "sd1.5" ]; then
                        export MODEL_PATH=$SD15_PATH
                    elif [ ${MODEL} == "sdxl" ]; then
                        export MODEL_PATH=$SDXL_PATH
                    elif [ ${MODEL} == "sd3" ]; then
                        export MODEL_PATH=$SD3_PATH
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
                        --num ${NUM} > "${result_dir_path}/unit_test.log" 2>&1

                    # Copy logs
                    cp ./outputs/gpu_worker_* "${result_dir_path}/"
                    rm -f ./outputs/gpu_worker_*
                done
            done
        done
    done
done 