import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("/workspace/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompts = [
    "A vast desert landscape bathed in the soft light of two moons, with a lone cactus reaching towards the sky. Style: Surreal, high detail.",
    "A neon-lit alleyway in a cyberpunk city, with rain reflecting the lights and a lone figure walking down the street. Style: Photorealistic, dark and moody.",
    "A vast library built within a coral reef, with schools of fish swimming between the glowing bookshelves. Style: Fantasy, vibrant colors.",
    "A majestic steampunk airship soaring through a cloudy sky, with gears and pistons visible on its hull. Style: Detailed, painterly.",
    "A fluffy cat wearing a spacesuit, floating in zero gravity with a playful grin on its face. Style: Cartoon, humorous.",
    "A close-up of a sparkling crystal, with tiny rainbows refracting in its facets and dewdrops clinging to its surface. Style: Photorealistic, high magnification.",
    "A towering art deco skyscraper bathed in the golden light of sunrise, with geometric patterns adorning its facade. Style: Architectural, detailed.",
    "A sleek and stylish robot posing on a runway, wearing a high-fashion outfit and futuristic accessories. Style: Digital art, futuristic.",
    "A mystical forest at night, with bioluminescent mushrooms illuminating the path and fireflies dancing in the air. Style: Fantasy, ethereal.",
    "A Roman gladiator wielding a neon lightsaber in a futuristic colosseum, battling a cyborg centaur. Style: Pop art, unexpected combination.",]

images = pipe(
    prompt=prompts,
    negative_prompt=[""] * len(prompts),
    num_inference_steps=40,
    guidance_scale=4.5,
).images

for i, img in enumerate(images):
    img.save(f"img{i}.png")