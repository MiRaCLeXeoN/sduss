import argparse
import aiohttp
import asyncio
import json
import requests
import random
import datetime

from typing import Iterable, List

def post_http_request(
        api_url: str,
        prompt: str,
        **kwargs
    ) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        **kwargs,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


async def get_image_from_session(
    session: aiohttp.ClientSession,
    id: int,
    api_url: str,
    prompt: str,
    **kwargs,
):
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        **kwargs,
    }
    start_time = datetime.datetime.now()
    async with session.post(url=api_url, headers=headers, json=pload) as response:
        end_time = datetime.datetime.now()
        img = await response.read()
        path = f"./outputs/imgs/api_client_{id}.png"
        with open(path, "wb") as f:
            f.write(img)
        print(f"{start_time=}, {end_time=}. Image saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/generate"

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

    resolutions = [256, 512, 768]

    async def main():
        async with aiohttp.ClientSession() as session:
            coros: List[asyncio.Future] = []
            for i, prompt in enumerate(prompts):
                resoluition = resolutions[random.randint(0, 2)]
                steps = random.randint(40, 60)
                print(f"{resoluition=}, {steps=}, prompt: {prompt!r}\n", flush=True)
                coros.append(get_image_from_session(
                    session=session,
                    id=i,
                    api_url=api_url,
                    prompt=prompt,
                    resolution=resoluition,
                    num_inference_steps=steps,
                ))
            await asyncio.gather(*coros)
    
    asyncio.get_event_loop().run_until_complete(main())