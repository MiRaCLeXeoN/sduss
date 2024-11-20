"""Client used for testing server."""
import argparse
import aiohttp
import datetime
import pandas
import asyncio
import logging
import os
import requests
import time

from typing import List

logger = logging.getLogger("client")

class Metrics:
    def __init__(self):
        self.order = [
            "index",
            "success",
            "request_id",
            "resolution",
            "step",
            "delay_time",
            "start_time",
            "end_time",
            "time_consumption",
            "prompt",
        ]
        self.header = ",".join(self.order)
    def get_str(
        self,
        **kwargs,
    ):
        tmp_strs = []
        for name in self.order:
            if name == "prompt":
                tmp_strs.append(f"\"{kwargs.pop(name)}\"")
            else:
                tmp_strs.append(f"{kwargs.pop(name)}")
        return ",".join(tmp_strs)
        

async def get_image_from_session(
    session: aiohttp.ClientSession,
    index: int,
    delay_time: float,
    api_url: str,
    prompt: str,
    resolution: int,
    num_inference_steps: int,
    **kwargs,
):
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "resolution": str(resolution),
        "num_inference_steps": str(num_inference_steps),
        **kwargs,
    }
    await asyncio.sleep(delay_time / 1000)
    print(f"start request {index}")
    start_time = datetime.datetime.now()
    try:
        async with session.post(url=api_url, headers=headers, json=pload, timeout=3600) as response:
            img = await response.read()
            end_time = datetime.datetime.now()

            if response.status == 200:
                img_name = response.headers.get("image_name")
                is_finished = response.headers.get("is_finished")
                path = result_dir_path +  f"/imgs/client_" + img_name
                # with open(path, "wb") as f:
                #     f.write(img)
                logger.info(metric.get_str(
                    index=index,
                    request_id=os.path.splitext(img_name)[0],
                    resolution=resolution,
                    step=num_inference_steps,
                    delay_time=delay_time,
                    start_time=start_time,
                    end_time=end_time,
                    time_consumption=(end_time - start_time).total_seconds(),
                    prompt=prompt,
                    success=is_finished
                ))
                global success_counter
                success_counter += 1
            else:
                logger.info(metric.get_str(
                    index=index,
                    request_id="None",
                    resolution=resolution,
                    step=num_inference_steps,
                    delay_time=delay_time,
                    start_time=start_time,
                    end_time=end_time,
                    time_consumption=(end_time - start_time).total_seconds(),
                    prompt=prompt,
                    success=False
                ))
        print(f"finish request {index}")
    except Exception as exc:
        print(exc)
        logger.info(metric.get_str(
                    index=index,
                    request_id="None",
                    resolution=resolution,
                    step=num_inference_steps,
                    delay_time=delay_time,
                    start_time=start_time,
                    end_time=datetime.datetime.now(),
                    time_consumption=None,
                    prompt=prompt,
                    success=False,
                ))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["sd1.5", "sdxl"],
    )
    parser.add_argument(
        "--qps",
    )
    parser.add_argument(
        "--distribution",
        choices=['equal', 'small', 'mid', 'large']
    )
    parser.add_argument(
        "--num",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--arrival_distri",
        default="gamma",
        type=str,
    )
    parser.add_argument(
        "--SLO",
        type=int,
    )
    parser.add_argument(
        "--policy",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    model = args.model
    qps = args.qps
    distribution = args.distribution
    slo = args.SLO
    policy = args.policy

    # Create result dir
    result_dir_path = f"./results/{model}/{args.arrival_distri}/{distribution}_{qps}_{slo}_{policy}"
    os.makedirs(result_dir_path + "/imgs", exist_ok=True)

    metric = Metrics()
    success_counter = 0

    log_file_name = result_dir_path + "/client.csv"
    handler = logging.FileHandler(log_file_name, mode="w")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info(metric.header)

    base_url = f"http://{args.host}:{args.port}/"

    max_retry = 60
    retry = max_retry
    while True:
        try:
            response = requests.get(url=base_url + "health", timeout=5)
            if response.status_code == 200:
                break
        except Exception as e:
            print(e)
            print("Connection time out, retry")
            time.sleep(5)
        retry -= 1
        if retry == 0: 
            break
    if not retry:
        print("Cannot connect to server, exit.")
        exit(-1)

    api_url = base_url + "generate"

    # Load data
    time_csv_path = f"./exp/{args.model}/{args.arrival_distri}/distri_{args.distribution}/qps_{args.qps}.csv"
    prompt_csv_path = f"./exp/0000.csv"
    time_csv = pandas.read_csv(time_csv_path)
    prompt_csv = pandas.read_csv(prompt_csv_path)

    async def main():
        async with aiohttp.ClientSession() as session:
            coros: List[asyncio.Future] = []
            for i in range(args.num):
                delay_time = time_csv.iloc[i, 0]
                resoluition = time_csv.iloc[i, 1]
                prompt = prompt_csv.iloc[i, 1]
                steps = time_csv.iloc[i, 2] if policy == "orca_resbyres" else 50
                coros.append(get_image_from_session(
                    session=session,
                    index=i,
                    api_url=api_url,
                    prompt=prompt,
                    resolution=resoluition,
                    num_inference_steps=steps,
                    delay_time=delay_time,
                ))
            await asyncio.gather(*coros)
    
    asyncio.get_event_loop().run_until_complete(main())

    # logger.info("---")
    # logger.info(f"Successful requests: {success_counter} / {args.num}")

    print("start server clear")
    requests.post(url=base_url + f"clear")
    print("finish server clear")