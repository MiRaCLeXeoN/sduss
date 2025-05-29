set -e

export MODEL="sd3"
# export NUM=""
export NUM="--num 500"

DP_LIST="1"
SDXL_QPS="0.1 0.2 0.3 0.4 0.5"
# POLICY_LIST="esymred fcfs_mixed orca_resbyres"
POLICY_LIST=" esymred "

if [[ $MODEL == "sd3" ]]; then
    export SLO="5"
elif [[ $MODEL == "sdxl" ]]; then
    export SLO="5"
fi

for dp_size in $DP_LIST; do
    for policy_name in $POLICY_LIST; do 
        for qps in $SDXL_QPS; do
            export QPS=$(awk "BEGIN {printf \"%.1f\", $qps * $dp_size}")
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

            echo "Start ${result_dir_path}, ${GPUS}"
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
                --data_parallel_size ${DATA_PARALLEL_SIZE} ${NUM}

            ps aux | grep sduss | grep -v grep | awk '{print $2}' | xargs kill -9

            # Copy logs
            cp ./outputs/gpu_worker_* "${result_dir_path}/"
            rm -f ./outputs/gpu_worker_*

            cp -r "${result_dir_path}" /workspace/results/${MODEL}/
            # rm -rf "${result_dir_path}"
        done
    done
done