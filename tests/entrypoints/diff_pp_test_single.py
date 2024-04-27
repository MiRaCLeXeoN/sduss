import PIL.Image
import torch
import PIL

from sduss import DiffusionPipeline

# 1. Create pipeline
pipe = DiffusionPipeline(model_name_or_pth="/data/home/zzp/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9",
                         torch_dtype=torch.float16)

sampling_params_cls = pipe.get_sampling_params_cls()

print(sampling_params_cls)

sampling_params = []

sampling_params.append(sampling_params_cls(prompt="a gigantic flower", num_inference_steps=50))
# sampling_params.append(sampling_params_cls(prompt="a flowring sitting on the crest."))

outputs = pipe.generate(sampling_params)

for i, output in enumerate(outputs):
    save_pth = f"./outputs/imgs/{i}.png"
    print(f"saving image from request {output.request_id} to {save_pth}\n")
    print(f"time consumption={output.time_consumption}")
    if isinstance(output.output.images, PIL.Image.Image):
        output.output.images.save(save_pth)
    else:
        print("Saving failed.")

