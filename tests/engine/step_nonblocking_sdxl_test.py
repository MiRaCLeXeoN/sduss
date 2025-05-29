import PIL.Image
import torch
import PIL
import random

from sduss import DiffusionPipeline

# 1. Create pipeline
pipe = DiffusionPipeline(model_name_or_pth="/data/home/zzp/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
                         policy="fcfs_mixed",
                         use_esymred=True,
                         use_mixed_precision=True,
                         disable_log_status=False,
                         max_batchsize=16,
                         non_blocking_step=True,
                         overlap_prepare=True,
                         torch_dtype=torch.float16)

sampling_params_cls = pipe.get_sampling_params_cls()

prompts = [
    "A vast desert landscape bathed in the soft light of two moons, with a lone cactus reaching towards the sky. Style: Surreal, high detail.",
    "A neon-lit alleyway in a cyberpunk city, with rain reflecting the lights and a lone figure walking down the street. Style: Photorealistic, dark and moody.",
    "A vast library built within a coral reef, with schools of fish swimming between the glowing bookshelves. Style: Fantasy, vibrant colors.",
    "A majestic steampunk airship soaring through a cloudy sky, with gears and pistons visible on its hull. Style: Detailed, painterly.",
    "A fluffy cat wearing a spacesuit, floating in zero gravity with a playful grin on its face. Style: Cartoon, humorous.",
    "A close-up of a sparkling crystal, with tiny rainbows refracting in its facets and dewdrops clinging to its surface. Style: Photorealistic, high magnification.",
    "A grand library built within a sunken city, sunlight filtering through cracks in the ruins, with schools of fish weaving between towering bookshelves.",
    "A cascading waterfall hidden behind a bustling city street, accessible only through a secret alleyway.",
    "A mystical forest at night, with bioluminescent mushrooms illuminating the path and fireflies dancing in the air. Style: Fantasy, ethereal.",
    "A Roman gladiator wielding a neon lightsaber in a futuristic colosseum, battling a cyborg centaur. Style: Pop art, unexpected combination.",]
prompts = prompts * 5

resolutions = [512, 768, 1024]

sampling_params = []

for i, prompt in enumerate(prompts):
    # step = random.randint(30, 50)
    step = 50
    print(f"{i=}, {step=}")
    sampling_params.append(sampling_params_cls(prompt=prompt, num_inference_steps=step,
                                               resolution=resolutions[random.randint(0, 2)]))


outputs = pipe.generate(sampling_params)

for i, output in enumerate(outputs):
    save_pth = f"./outputs/imgs/{i}.png"
    print(f"saving image from request {output.request_id} to {save_pth}\n")
    print(f"time consumption={output.time_consumption}")
    if isinstance(output.output.images, PIL.Image.Image):
        output.output.images.save(save_pth)
    else:
        print("Saving failed.")
