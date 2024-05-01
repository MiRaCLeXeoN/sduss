import PIL.Image
import torch
import PIL
import random

from sduss import DiffusionPipeline

# 1. Create pipeline
pipe = DiffusionPipeline(model_name_or_pth="/home/zzp/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
                         policy="fcfs_mixed",
                         use_esymred=True,
                         use_mixed_precision=True,
                         disable_log_status=False,
                         torch_dtype=torch.float16)

sampling_params_cls = pipe.get_sampling_params_cls()

prompt = "A vast desert landscape bathed in the soft light of two moons, with a lone cactus reaching towards the sky. Style: Surreal, high detail."

resolutions = [256, 512, 768]

sampling_params = []

sampling_params.append(sampling_params_cls(prompt=prompt, num_inference_steps=50,
                                            resolution=resolutions[2]))


outputs = pipe.generate(sampling_params)

for i, output in enumerate(outputs):
    save_pth = f"./outputs/imgs/{i}.png"
    print(f"saving image from request {output.request_id} to {save_pth}\n")
    print(f"time consumption={output.time_consumption}")
    if isinstance(output.output.images, PIL.Image.Image):
        output.output.images.save(save_pth)
    else:
        print("Saving failed.")

