# Mixfusion

## How to run

Clone the repo and run the following code to setup the environment

```
conda env create -f ./conda.yml
conda activate sduss
pip install -e .
```

You can run the scripts in [./scripts](./scripts/) to get the results presented in our paper.

Or you can run commands like

```
python ./sduss/entrypoints/api_server.py \
    --model_name_or_pth {MODEL_NAME_OR_PATH} \
    --policy esymred \
    --dispatcher_policy greedy \
    --mixed_precision \
    --use_esymred \
    --max_batchsize 10 \
    --torch_dtype "float16" \
    --engine_use_mp \
    --worker_use_mp \
    --data_parallel_size 2 \
    --gpus [0-1]
```

to lunch a server and then send requests to it to start a job. 

Please replace the "MODEL_NAME_OR_PATH" with your model's path. Currently, we only support SDXL and SD3 from diffusers.