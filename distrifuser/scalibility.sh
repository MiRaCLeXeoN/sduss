set -e

DP_LIST="8 4 2"
MODELS=("sd3" "sdxl")

for dp_size in $DP_LIST; do
    for MODEL in "${MODELS[@]}"; do
        if [[ ${MODEL} == "sd3" ]]; then
            export QPS_LIST="0.1 0.2 0.3 0.4 0.5"
            export SLO="5"
        elif [[ ${MODEL} == "sdxl" ]]; then
            export QPS_LIST="0.8 0.9 1.0 1.1 1.2"
            export SLO="5"
        fi

        for qps in $QPS_LIST; do
            export MODEL=${MODEL}

            export QPS=$(awk "BEGIN {printf \"%.1f\", $qps * $dp_size}")
            export DATA_PARALLEL_SIZE=$dp_size

            result_dir_path="./results/${MODEL}/${QPS}_${SLO}_distrifusion_${DATA_PARALLEL_SIZE}"
            if [ -d "${result_dir_path}" ]; then
                find ${result_dir_path} -type f -delete
            else
            mkdir -p ${result_dir_path}
            fi

            echo "Start ${result_dir_path}"

            torchrun --nproc_per_node=${DATA_PARALLEL_SIZE} test.py > test.log 2>test.err

            # Copy logs
            # cp -r "${result_dir_path}" /workspace/results/${MODEL}/
            cp test.log "${result_dir_path}/test.log"
            cp test.err "${result_dir_path}/test.err"
        done
    done
done