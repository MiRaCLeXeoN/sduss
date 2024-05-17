# export USE_MIXED_PRECISION="--use_mixed_precision"
# export OVERLAP_PREPARE="--overlap_prepare"
# export NON_BLOCKING_STEP="--non_blocking_step"

export USE_MIXED_PRECISION=""
export OVERLAP_PREPARE=""
export NON_BLOCKING_STEP=""

export SLO="5"
export DISTRIBUTION="equal"
export NUM=100

export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

SD15_QPS="0.45 0.5 0.55 0.6 0.65 0.7 0.75"
SDXL_QPS="0.15 0.175 0.2 0.225 0.25 0.275 0.3"
POLICY_LIST="fcfs_single orca_resbyres orca_round_robin"

export MODEL="sd1.5"

set -e

echo "Start sd1.5 test in exp1"
for qps in $SD15_QPS; do
    for policy_name in $POLICY_LIST; do 
        if [[ $policy_name == "esymred" ]]; then
            export USE_MIXED_PRECISION="--use_mixed_precision"
            export OVERLAP_PREPARE="--overlap_prepare"
            export NON_BLOCKING_STEP="--non_blocking_step"
        else
            export USE_MIXED_PRECISION=""
            export OVERLAP_PREPARE=""
            export NON_BLOCKING_STEP=""
        fi
        export QPS=$qps
        export POLICY=$policy_name
        folder_path="./results/${MODEL}/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
        if [ -d "${folder_path}" ]; then
            find ${folder_path} -type f -delete
        fi
        sbatch_output=$(sbatch ./scripts/slurm/unit_test.slurm)
        job_num=${sbatch_output:0-4}
        echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION"
        # Wait until server is ready
        sleep 100
        /home/zzp/miniconda3/envs/sduss/bin/python ./tests/server/esymred_test.py \
            --model ${MODEL} \
            --qps ${QPS} \
            --distribution ${DISTRIBUTION} \
            --SLO ${SLO} \
            --policy ${POLICY} \
            --host hepnode2 \
            --port 8000 \
            --num $NUM
        $(scancel ${job_num})
        echo "cancelled job $job_num"
        cp ./outputs/*$job_num.* ${folder_path}/
        cp ./outputs/ray.log ${folder_path}/
    done
done


export MODEL="sdxl"
echo "Start $MODEL test in exp1"
for qps in $SDXL_QPS; do
    for policy_name in $POLICY_LIST; do 
        if [[ $policy_name == "esymred" ]]; then
            export USE_MIXED_PRECISION="--use_mixed_precision"
            export OVERLAP_PREPARE="--overlap_prepare"
            export NON_BLOCKING_STEP="--non_blocking_step"
        else
            export USE_MIXED_PRECISION=""
            export OVERLAP_PREPARE=""
            export NON_BLOCKING_STEP=""
        fi
        export QPS=$qps
        export POLICY=$policy_name
        folder_path="./results/${MODEL}/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
        if [ -d "${folder_path}" ]; then
            find ${folder_path} -type f -delete
        fi
        sbatch_output=$(sbatch ./scripts/slurm/unit_test.slurm)
        job_num=${sbatch_output:0-4}
        echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION"
        # Wait until server is ready
        sleep 100
        /home/zzp/miniconda3/envs/sduss/bin/python ./tests/server/esymred_test.py \
            --model ${MODEL} \
            --qps ${QPS} \
            --distribution ${DISTRIBUTION} \
            --SLO ${SLO} \
            --policy ${POLICY} \
            --host hepnode2 \
            --port 8000 \
            --num $NUM
        $(scancel ${job_num})
        echo "cancelled job $job_num"
        cp ./outputs/*$job_num.* ${folder_path}/
        cp ./outputs/ray.log ${folder_path}/
    done
done