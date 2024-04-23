import torch

from sduss import DiffusionPipeline

# 1. Create pipeline
pipe = DiffusionPipeline(model_name_or_pth="/data/home/zzp/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9",
                         torch_dtype=torch.float16)

sampling_params_cls = pipe.get_sampling_params_cls()

print(sampling_params_cls)

sampling_params = []

sampling_params.append(sampling_params_cls(prompt="astrount riding a horse on the moon."))
sampling_params.append(sampling_params_cls(prompt="a flowring sitting on the crest."))

outputs = pipe.generate(sampling_params)
