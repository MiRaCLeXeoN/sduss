export MODEL="sd3"
export NUM=200

if [[ $MODEL == "sd3" ]]; then
    export SLO="5"
elif [[ $MODEL == "sdxl" ]]; then
    export SLO="3"
fi

qps=0.5
dp_size=1
export GPUS="[7]"
export DATA_PARALLEL_SIZE=${dp_size}
export QPS=$(awk "BEGIN {printf \"%.1f\", $qps * $dp_size}")

#export POLICY="orca_resbyres"
export POLICY="fcfs_mixed"
# export POLICY="esymred"

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
bash ./scripts/h100/unit_test.sh > unit_test.log 2>&1 &
# echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION, SLO=$SLO"
echo "Start ${result_dir_path}"
# Wait until server is ready
sleep 45
python ./tests/server/esymred_test.py \
    --model ${MODEL} \
    --qps ${QPS} \
    --SLO ${SLO} \
    --policy ${POLICY} \
    --host localhost \
    --port 8000 \
    --data_parallel_size ${DATA_PARALLEL_SIZE} \
    --num ${NUM}

ps aux | grep sduss | grep -v grep | awk '{print $2}' | xargs kill -9
# echo "cancelled job $job_num"
# cp -r "${result_dir_path}" /workspace/results/${MODEL}/
