export DATA_PARALLEL_SIZE=2
export SLO="3"
export MODEL="sd3"
export QPS="1.2"
export GPUS="[6-7]"
export NUM=50

# export POLICY="orca_resbyres"
export POLICY="fcfs_mixed"
# export POLICY="esymred"

export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

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
# cp ./outputs/*$job_num.* ${result_dir_path}/