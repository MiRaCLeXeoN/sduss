conda activate distrifuser

source ../scripts/env.sh

# Figure 13
bash ./distribution.sh

# Figure 14 and 15
bash ./scalibility.sh

# copy results to main results folder
cp -r ./results/* ../results/