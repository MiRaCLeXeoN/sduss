# Mixfusion

## Hardware

Our project is based on NVIDIA H100 GPUs. Since the predictor (a small model) is trained based on the data collected on H100, the system can only perform normally on H100 machines. So a 8xH100 machine is required to reproduce the results.

## Environment

First download the docker image with appropriate CUDA version.

To avoid re-downloding the model weights inside the container, you can reuse the model weights on your host machine if there is. Just replace the `<huggingace path>` below to the PARENT path of huggingface root path. (Inside this path, there should be a `hub` directory containing the models), like: 

```
$ ls huggingface/hub/
models--stabilityai--stable-diffusion-3.5-medium  models--stabilityai--stable-diffusion-xl-base-1.0  version_diffusers_cache.txt  version.txt
```

Two model weights are required: `models--stabilityai--stable-diffusion-3.5-medium` and `models--stabilityai--stable-diffusion-xl-base-1.0`. You can find both on huggingface. Follow the instructions on huggingface to download the weights.

Run the docker containers. If you choose to download the model inside the container, then you don't need to impose `-v <>:<>` option.

```
docker run -d --name mixfusion --gpus all -v <huggingface path>:/workspace -it nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 bash
```

Attach to the container.

```
docker exec -it mixfusion bash
```

Then install the conda environment. (Remember to refresh your bash after installation)

```
cd ~
apt-get update
apt-get install -y wget git vim
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Clone this repo into a specific path. Here we use `/root` directory. 


And then we install the conda env. (This env name is sduss, not mixfusion!!!)

```
cd <repo path>
conda env create -f conda.yml
conda env activate sduss
```

`cuml-cu12` is commented out in `conda.yml`, since we find it sometimes trouble-raising. So we have to install it manually.

```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cuml-cu12==24.8.0"
```

Install this package.

```
pip install -e .
```

### Install env for distrifuser

We compare our project against distrifuser. To achieve this goal, we made some modifications to the code and integrated their code here. Since it has different dependencies from ours, you should install a separate conda env for it.

```
cd distrifuser
conda env create -f distrifuser.yml
```

## Run

Before running tests, you `MUST` update the paths inside `./scripts/env.sh`. Instructions are embedded there.

Now we can run all the exp.

```
bash ./scripts/paper/run_all.sh
```

This script automatically runs results for Figure 12, 13, 14, and 15 presented in our paper.

It takes roughly 25 hours to finish all the experiments. You can check the `run_all.sh` to see the guidance of adjusting experiment time.

### Run results of distrifuser

```
cd distrifuser
conda activate distrifuser
bash ./run_all.sh
```