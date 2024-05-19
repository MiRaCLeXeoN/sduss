export ESYMRED_PREDICTOR_PATH="./exp/$MODEL.pkl"
export ESYMRED_EXEC_TIME_DIR="./exp/profile"

export DISTRIBUTION="equal"
export POLICY="esymred"
export NUM=100

SD15_QPS="0.3 0.375 0.45"
SDXL_QPS="0.1 0.175 0.25"
SLO_LIST="3 5 7"

set -e

# Do sd1.5
export MODEL="sd1.5"
for slo in $SLO_LIST; do
    for qps in $SD15_QPS; do
        export SLO=$slo
        export QPS=$qps

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
        folder_path="./results/${MODEL}/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
        if [ -d "${folder_path}" ]; then
            find ${folder_path} -type f -delete
        fi
        sbatch_output=$(sbatch ./scripts/slurm/unit_test.slurm)
        job_num=${sbatch_output:0-4}
        echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION, SLO=$SLO"
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
        sleep 3
        $(scancel ${job_num})
        echo "cancelled job $job_num"
        cp ./outputs/*$job_num.* ${folder_path}/
    done
done


# Do sdxl
export MODEL="sdxl"
for slo in $SLO_LIST; do
    for qps in $SDXL_QPS; do
        export SLO=$slo
        export QPS=$qps

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
        folder_path="./results/${MODEL}/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
        if [ -d "${folder_path}" ]; then
            find ${folder_path} -type f -delete
        fi
        sbatch_output=$(sbatch ./scripts/slurm/unit_test.slurm)
        job_num=${sbatch_output:0-4}
        echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION, SLO=$SLO"
        # Wait until server is ready
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
        sleep 3
        $(scancel ${job_num})
        echo "cancelled job $job_num"
        cp ./outputs/*$job_num.* ${folder_path}/
    done
done