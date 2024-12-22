import argparse
import json
import requests
import random

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
    print(response)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/generate"

    prompts = [
    "A vast desert landscape bathed in the soft light of two moons, with a lone cactus reaching towards the sky. Style: Surreal, high detail."]

    resolutions = [256, 512, 768]

    for i, prompt in enumerate(prompts):
        print(f"Prompt: {prompt!r}\n", flush=True)
        response = post_http_request(api_url, prompt=prompt,
                                     resolution=resolutions[random.randint(0, 2)],
                                     num_inference_steps=random.randint(40, 60))

        img = response.content

        path = f"./outputs/imgs/api_client_{i}.png"
        with open(path, "wb") as f:
            f.write(img)