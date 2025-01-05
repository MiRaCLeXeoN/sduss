export DATA_PARALLEL_SIZE=8
export SLO="3"
export MODEL="sdxl"

export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

set -e

DP_LIST="1 2 4 8"
SDXL_QPS="0.6 0.7 0.8 0.9 1"
POLICY_LIST="esymred fcfs_mixed orca_resbyres"

export MODEL="sdxl"
for policy_name in $POLICY_LIST; do 
    for qps in $SDXL_QPS; do
        for dp_size in $DP_LIST; do
            export QPS=$(awk "BEGIN {printf \"%.1f\", $qps * $dp_size}")
            export POLICY=$policy_name
            export DATA_PARALLEL_SIZE=$dp_size
            last_gpu=$(awk "BEGIN {printf $dp_size - 1}")
            export GPUS="[0-${last_gpu}]"
            if [[ $POLICY == "esymred" || $POLICY == "fcfs_mixed" ]]; then
                export USE_MIXED_PRECISION="--use_mixed_precision"
                export ESYMRED_USE_CACHE="TRUE"
            elif [[ $POLICY == "orca_resbyres" ]]; then
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

            echo "Start ${result_dir_path}"
            bash ./scripts/h100/unit_test.sh > "${result_dir_path}/unit_test.log" 2>&1 &
            # Wait until server is ready
            sleep 60
            python ./tests/server/esymred_test.py \
                --model ${MODEL} \
                --qps ${QPS} \
                --SLO ${SLO} \
                --policy ${POLICY} \
                --host localhost \
                --port 8000 \
                --data_parallel_size ${DATA_PARALLEL_SIZE}

            ps aux | grep sduss | grep -v grep | awk '{print $2}' | xargs kill -9

            # Copy logs
            cp ./outputs/gpu_worker_* "${result_dir_path}/"
            rm -f ./outputs/gpu_worker_*
        done
    done
done