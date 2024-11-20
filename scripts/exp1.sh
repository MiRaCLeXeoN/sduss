export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

export SLO="5"
export DISTRIBUTION="equal"
export NUM=500

SD15_QPS="0.4 0.425 0.45 0.475 0.5"
SDXL_QPS="0.2 0.225 0.25 0.275 0.3"

# SD15_QPS="0.75 0.8"
# SDXL_QPS="0.25 0.275 0.3 0.325"
POLICY_LIST="esymred fcfs_mixed fcfs_single orca_resbyres"

set -e

export MODEL="sd1.5"
export ARRIVAL_DISTRI="gamma"

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
        folder_path="./results/${MODEL}/${ARRIVAL_DISTRI}/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
        if [ -d "${folder_path}" ]; then
            find ${folder_path} -type f -delete
        fi
        # sbatch_output=$(sbatch ./scripts/slurm/unit_test.slurm)
        bash ./scripts/h100/unit_test.sh > exp1.log 2>&1 &
        # job_num=${sbatch_output:0-4}
        # echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION, SLO=$SLO"
        # Wait until server is ready
        sleep 60
        python ./tests/server/esymred_test.py \
            --model ${MODEL} \
            --qps ${QPS} \
            --distribution ${DISTRIBUTION} \
            --arrival_distri ${ARRIVAL_DISTRI} \
            --SLO ${SLO} \
            --policy ${POLICY} \
            --host localhost \
            --port 8000 \
            --num $NUM
        sleep 2
        # $(scancel ${job_num})
        ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
        echo "cancelled job $job_num"
        # cp ./outputs/*$job_num.* ${folder_path}/
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
        folder_path="./results/${MODEL}/${ARRIVAL_DISTRI}/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
        if [ -d "${folder_path}" ]; then
            find ${folder_path} -type f -delete
        fi
        # sbatch_output=$(sbatch ./scripts/slurm/unit_test.slurm)
        # job_num=${sbatch_output:0-4}
        # echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION, SLO=$SLO"
        bash ./scripts/h100/unit_test.sh > exp1.log 2>&1 &
        # Wait until server is ready
        sleep 80
        python ./tests/server/esymred_test.py \
            --model ${MODEL} \
            --qps ${QPS} \
            --distribution ${DISTRIBUTION} \
            --arrival_distri ${ARRIVAL_DISTRI} \
            --SLO ${SLO} \
            --policy ${POLICY} \
            --host localhost \
            --port 8000 \
            --num $NUM
        sleep 2
        # $(scancel ${job_num})
        ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
        echo "cancelled job $job_num"
        # cp ./outputs/*$job_num.* ${folder_path}/
        sleep 2
    done
done