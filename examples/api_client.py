"""Example Python client for api_server"""

import argparse
import json
from typing import Iterable, List

import requests

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/generate"
    prompt = args.prompt

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url)

    img = response.content

    path = "./api_client.png"
    with open(path, "wb") as f:
        f.write(img)