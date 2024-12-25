# export USE_MIXED_PRECISION=""
# export OVERLAP_PREPARE=""
# export NON_BLOCKING_STEP=""
# export USE_MIXED_PRECISION="--use_mixed_precision"
# export OVERLAP_PREPARE="--overlap_prepare"
# export NON_BLOCKING_STEP="--non_blocking_step"

export DATA_PARALLEL_SIZE=2
export SLO="5"
export DISTRIBUTION="equal"
export NUM=100
export MODEL="sdxl"
export QPS="0.4"
export GPUS="[6, 7]"
ARRIVAL_DISTRI="gamma"

# export POLICY="fcfs_single"
export POLICY="esymred"

export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

if [[ $POLICY == "esymred" ]]; then
    export USE_MIXED_PRECISION="--use_mixed_precision"
    export OVERLAP_PREPARE="--overlap_prepare"
    export NON_BLOCKING_STEP="--non_blocking_step"
elif [[ $POLICY == "fcfs_mixed" ]]; then
    export USE_MIXED_PRECISION="--use_mixed_precision"
    export OVERLAP_PREPARE=""
    export NON_BLOCKING_STEP="--non_blocking_step"
else
    export USE_MIXED_PRECISION=""
    export OVERLAP_PREPARE=""
    export NON_BLOCKING_STEP="--non_blocking_step"
fi
folder_path="./results/${MODEL}/${ARRIVAL_DISTRI}/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
if [ -d "${folder_path}" ]; then
    find ${folder_path} -type f -delete
fi
bash ./scripts/h100/unit_test.sh > unit_test.log 2>&1 &
# echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION, SLO=$SLO"
# Wait until server is ready
sleep 60
python ./tests/server/esymred_test.py \
    --model ${MODEL} \
    --qps ${QPS} \
    --distribution ${DISTRIBUTION} \
    --SLO ${SLO} \
    --policy ${POLICY} \
    --host localhost \
    --port 8000 \
    --num $NUM
sleep 3
ps aux | grep sduss | grep -v grep | awk '{print $2}' | xargs kill -9
# echo "cancelled job $job_num"
cp ./outputs/*$job_num.* ${folder_path}/