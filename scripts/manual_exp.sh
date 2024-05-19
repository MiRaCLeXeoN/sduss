# export USE_MIXED_PRECISION=""
# export OVERLAP_PREPARE=""
# export NON_BLOCKING_STEP=""
# export USE_MIXED_PRECISION="--use_mixed_precision"
# export OVERLAP_PREPARE="--overlap_prepare"
# export NON_BLOCKING_STEP="--non_blocking_step"

export SLO="5"
export DISTRIBUTION="equal"
export NUM=100
export MODEL="sd1.5"
export QPS="0.35"

# export POLICY="fcfs_single"
export POLICY="esymred"

export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

if [[ $POLICY == "esymred" ]]; then
    export USE_MIXED_PRECISION="--use_mixed_precision"
    export OVERLAP_PREPARE="--overlap_prepare"
    export NON_BLOCKING_STEP="--non_blocking_step"
else
    export USE_MIXED_PRECISION=""
    export OVERLAP_PREPARE=""
    export NON_BLOCKING_STEP=""
fi
folder_path="./results/${MODEL}/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
if [ -d "${folder_path}" ]; then
    find ${folder_path} -type f -delete
fi
sbatch_output=$(sbatch ./scripts/slurm/unit_test.slurm)
job_num=${sbatch_output:0-4}
echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION"
# Wait until server is ready
sleep 60
python ./tests/server/esymred_test.py \
    --model ${MODEL} \
    --qps ${QPS} \
    --distribution ${DISTRIBUTION} \
    --SLO ${SLO} \
    --policy ${POLICY} \
    --host hepnode3 \
    --port 8000 \
    --num $NUM
sleep 3
$(scancel ${job_num})
echo "cancelled job $job_num"
cp ./outputs/*$job_num.* ${folder_path}/