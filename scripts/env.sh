# ----------------------------------
# To run the test successfully, you MUST update the following variables
# ----------------------------------
export TORCH_INCLUDE_PATH="/root/miniconda3/envs/sduss/lib/python3.9/site-packages/torch/include"

# 2. specify the model path.
# We use the direct path of the model weights downloaded from Hugging Face.
# The `workspace` folder is mounted from the host machine inside the docker container.
# If you choose to download the model weights inside the docker container,
# please make sure to update the following paths accordingly.
# Take the following paths as examples.
export SD15_PATH="/workspace/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1"
export SDXL_PATH="/workspace/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
export SD3_PATH="/workspace/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"

# 3. This variable controls the number of requests to be sent in each experiment.
# We use 500 for default settings. But this would cause the experiments to run for a long time.
# 500 requests may take around 20-30 hours to finish all experiments.
# The smaller the number is, the faster the experiments finish.
export NUM="500"