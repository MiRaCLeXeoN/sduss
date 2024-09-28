export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

export SLO="5"
export DISTRIBUTION="equal"
export NUM=100

# SD15_QPS="0.3 0.325 0.35 0.375 0.4 0.425 0.45"
# SDXL_QPS="0.1 0.125 0.15 0.175 0.2 0.225 0.25"
SD15_QPS="1.1 1.2 1.3 1.4 1.5"
SDXL_QPS="0.5 0.55 0.6 0.65 0.7"
POLICY_LIST="esymred fcfs_mixed fcfs_single orca_resbyres"

set -e

export MODEL="sd1.5"
echo "Start $MODEL test in exp1"
for qps in $SD15_QPS; do
    for policy_name in $POLICY_LIST; do 
        export QPS=$qps
        export POLICY=$policy_name
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
        folder_path="./results/${MODEL}/uniform/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
        if [ -d "${folder_path}" ]; then
            find ${folder_path} -type f -delete
        fi
        bash ./scripts/bash/unit_test.bash
        # job_num=${sbatch_output:0-4}
        pid=$!
        # echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION, SLO=$SLO"
        # Wait until server is ready
        sleep 60
        python ./tests/server/esymred_test.py \
            --model ${MODEL} \
            --qps ${QPS} \
            --distribution ${DISTRIBUTION} \
            --SLO ${SLO} \
            --policy ${POLICY} \
            --host hepnode2 \
            --port 8000 \
            --num $NUM
        sleep 2
        # $(scancel ${job_num})
        kill $pid
        echo "cancelled job $job_num"
        cp ./outputs/*$job_num.* ${folder_path}/
        sleep 2
    done
done


export MODEL="sdxl"
echo "Start $MODEL test in exp1"
for qps in $SDXL_QPS; do
    for policy_name in $POLICY_LIST; do 
        export QPS=$qps
        export POLICY=$policy_name
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
        folder_path="./results/${MODEL}/uniform/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
        if [ -d "${folder_path}" ]; then
            find ${folder_path} -type f -delete
        fi
        # sbatch_output=$(sbatch ./scripts/slurm/unit_test.slurm)
        # job_num=${sbatch_output:0-4}
        # echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION, SLO=$SLO"
        # Wait until server is ready
        bash ./scripts/bash/unit_test.bash
        # job_num=${sbatch_output:0-4}
        pid=$!
        sleep 80
        python ./tests/server/esymred_test.py \
            --model ${MODEL} \
            --qps ${QPS} \
            --distribution ${DISTRIBUTION} \
            --SLO ${SLO} \
            --policy ${POLICY} \
            --host hepnode2 \
            --port 8000 \
            --num $NUM
        sleep 2
        # $(scancel ${job_num})
        kill $pid
        echo "cancelled job $job_num"
        cp ./outputs/*$job_num.* ${folder_path}/
        sleep 2
    done
done