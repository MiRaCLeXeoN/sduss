# export USE_MIXED_PRECISION="--use_mixed_precision"
# export OVERLAP_PREPARE="--overlap_prepare"
# export NON_BLOCKING_STEP="--non_blocking_step"

export USE_MIXED_PRECISION=""
export OVERLAP_PREPARE=""
export NON_BLOCKING_STEP=""

export SLO="5"
export NUM=100

SD15_QPS="0.6"
SDXL_QPS="0.225"
# No esymred
POLICY_LIST="fcfs_single orca_resbyres orca_round_robin"
DISTRIBUTION_LIST="small mid large"
MODEL_LIST="sd1.5 sdxl"

set -e

for MODEL in $MODEL_LIST; do
    echo "Start $MODEL test in exp3"
    for DISTRIBUTION in $DISTRIBUTION_LIST; do
        for policy_name in $POLICY_LIST; do 
            if [ $MODEL == "sd1.5" ]; then
                export QPS=$SD15_QPS
            else
                export QPS=$SDXL_QPS
            fi
            export MODEL=$MODEL
            export POLICY=$policy_name
            export DISTRIBUTION=$DISTRIBUTION
            sbatch_output=$(sbatch ./scripts/slurm/unit_test.slurm)
            job_num=${sbatch_output:0-4}
            echo "Got job $job_num to run model=$MODEL, qps=$QPS, policy=$POLICY, distribution=$DISTRIBUTION"
            # Wait until server is ready
            sleep 100
            folder_path="./results/${MODEL}/${DISTRIBUTION}_${QPS}_${SLO}_${POLICY}"
            if [ -d "${folder_path}" ]; then
                find ${folder_path} -type f -delete
            fi
            /home/zzp/miniconda3/envs/sduss/bin/python ./tests/server/esymred_test.py \
                --model ${MODEL} \
                --qps ${QPS} \
                --distribution ${DISTRIBUTION} \
                --SLO ${SLO} \
                --policy ${POLICY} \
                --host hepnode3 \
                --port 8000 \
                --num $NUM
            $(scancel ${job_num})
            echo "cancelled job $job_num"
            cp ./outputs/*$job_num.* ${folder_path}/
            sleep 5
        done
    done
done