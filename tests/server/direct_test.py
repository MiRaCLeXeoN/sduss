"""Client used for testing server."""
import argparse
import datetime
import pandas
import asyncio
import logging
import os
import uuid

from typing import List

import torch.multiprocessing as mp

from sduss.engine.async_engine import AsyncEngine
from sduss.engine.arg_utils import AsyncEngineArgs
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.model_executor.diffusers import BasePipeline
from sduss.model_executor.model_loader import get_pipeline_cls

logger = logging.getLogger("client")

engine = None
sampling_param_cls = None

def random_uuid() -> int:
    return uuid.uuid4().int

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
        

async def get_image_from_engine(
    index: int,
    delay_time: float,
    prompt: str,
    resolution: int,
    num_inference_steps: int,
    **kwargs,
):
    params = {
        "prompt": prompt,
        "resolution": str(resolution),
        "num_inference_steps": str(num_inference_steps),
        **kwargs,
    }
    await asyncio.sleep(delay_time / 1000)
    # print(f"start request {index}")
    start_time = datetime.datetime.now()
    try:
        global engine, sampling_param_cls
        request_id = random_uuid()
        sampling_params = sampling_param_cls(**params)
        results_generator = engine.generate(request_id=request_id, sampling_params=sampling_params)

        async for request_output in results_generator:
            final_output = request_output
        assert final_output is not None

        end_time = datetime.datetime.now()

        logger.info(metric.get_str(
            index=index,
            request_id=request_id,
            resolution=resolution,
            step=num_inference_steps,
            delay_time=delay_time,
            start_time=start_time,
            end_time=end_time,
            time_consumption=(end_time - start_time).total_seconds(),
            prompt=prompt,
            success=final_output,
        ))

        # print(f"finish request {index}")
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

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["sd1.5", "sdxl", "sd3"],
    )
    parser.add_argument(
        "--qps",
        type=str,
    )
    parser.add_argument(
        "--num",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--SLO",
        type=int,
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    parser = AsyncEngineArgs.add_args_to_parser(parser)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = handle_args()

    model = args.__dict__.pop("model")
    qps = args.__dict__.pop("qps")
    slo = args.__dict__.pop("SLO")
    num_reqs = args.__dict__.pop("num")
    host = args.__dict__.pop("host")
    port = args.__dict__.pop("port")
    # Don't pop these, it will be used in AsyncEngineArgs
    data_parallel_size = args.data_parallel_size
    policy = args.policy

    mp.set_start_method("spawn")

    # Create result dir
    result_dir_path = f"./results/{model}/{qps}_{slo}_{policy}_{data_parallel_size}"
    os.makedirs(result_dir_path + "/imgs", exist_ok=True)

    metric = Metrics()

    log_file_name = result_dir_path + "/client.csv"
    handler = logging.FileHandler(log_file_name, mode="w")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info(metric.header)

    # Load data
    time_csv_path = f"./exp/{model}/qps_{qps}.csv"
    prompt_csv_path = f"./exp/0000.csv"
    time_csv = pandas.read_csv(time_csv_path)
    prompt_csv = pandas.read_csv(prompt_csv_path)

    # setup engine
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncEngine.from_engine_args(engine_args)

    pipeline_cls: BasePipeline = get_pipeline_cls(engine.pipeline_config)
    sampling_param_cls: BaseSamplingParams = pipeline_cls.get_sampling_params_cls()

    async def main():
        coros: List[asyncio.Future] = []
        if num_reqs is None:
            num = len(time_csv)
        else:
            num = num_reqs
        for i in range(num):
            delay_time = time_csv.iloc[i, 0]
            resoluition = time_csv.iloc[i, 1]
            prompt = prompt_csv.iloc[i, 1]
            # steps = time_csv.iloc[i, 2] if policy == "orca_resbyres" else 50
            if policy == "orca_resbyres":
                steps = time_csv.iloc[i, 2]
            else:
                steps = 50
            coros.append(get_image_from_engine(
                engine=engine,
                index=i,
                delay_time=delay_time,
                prompt=prompt,
                resolution=resoluition,
                num_inference_steps=steps,
            ))
        await asyncio.gather(*coros)
    
    asyncio.get_event_loop().run_until_complete(main())

    print("start server clear")
    asyncio.get_event_loop().run_until_complete(engine.clear())
    print("finish server clear")