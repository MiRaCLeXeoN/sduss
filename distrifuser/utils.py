import time
import uuid
import os

import torch
import pandas as pd

from typing import Dict, List, Tuple

from distrifuser.pipelines import DistriSDXLPipeline, DistriSD3Pipeline
from distrifuser.utils import DistriConfig

sd3_path = os.environ.get("SD3_PATH", None)
sdxl_path = os.environ.get("SDXL_PATH", None)

if sd3_path is None:
    raise ValueError("Please set SD3_PATH environment variable")
if sdxl_path is None:
    raise ValueError("Please set SDXL_PATH environment variable")

setup = {
    "sd3": {
        "model": sd3_path,
        "pipeline_class": DistriSD3Pipeline,
        "max_bs": {
            2: {
                1024: 8,
                768: 16,
                512: 40,
            },
            4: {
                1024: 8,
                768: 16,
                512: 40,
            },
            8: {
                1024: 8,
                768: 16,
                512: 40,
            },
        },
    },
    "sdxl": {
        "model": sdxl_path,
        "pipeline_class": DistriSDXLPipeline,
        "max_bs": {
            2: {
                1024: 12,
                768: 20,
                512: 40,
            },
            4: {
                1024: 12,
                768: 20,
                512: 40,
            },
            8: {
                1024: 12,
                768: 20,
                512: 40,
            },
        },
    },
}


def get_pp(pipeline_cls, model_path, res, re_init):
    distri_config = DistriConfig(
        height=res,
        width=res,
        warmup_steps=4,
        use_cuda_graph=False,
        re_init=re_init,
        split_batch=False,
        do_classifier_free_guidance=True,
    )
    pipeline = pipeline_cls.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline.set_progress_bar_config(disable=True)
    return (distri_config, pipeline)


def get_pipeline(model):
    # res = [512, 768, 1024]
    res = [512]
    pipelines = {}
    pipeline_cls = setup[model]["pipeline_class"]
    model_path = setup[model]["model"]
    for i, r in enumerate(res):
        re_init = False if i == 0 else True
        # re_init = True
        pipelines[r] = get_pp(pipeline_cls, model_path, r, re_init=re_init)
        torch.cuda.synchronize()
        # print(f"get pipeline for {r=}")
    return pipelines[512]


def get_csv_files(model, qps: str):
    prompt_csv_path = f"./exp/0000.csv"
    prompt_csv = pd.read_csv(prompt_csv_path)
    qps_csv_path = f"./exp/{model}/qps_{qps}.csv"
    qps_csv = pd.read_csv(qps_csv_path)
    return prompt_csv, qps_csv


class Request:
    def __init__(self, index, resolution, step, prompt, delay_time=0.0):
        self.index = index
        self.resolution = resolution
        self.step = step
        self.prompt = prompt
        self.delay_time = delay_time

        self.request_id = uuid.uuid4().int
        self.success = False
        self.start_time = 0.0
        self.end_time = 0.0
        self.time_consumption = 0.0


class RequestPool:
    def __init__(self, model, prompt_csv, qps_csv, init_time, world_size):
        self.resolutions = [512, 768, 1024]
        self.pool: Dict[int, List[Request]] = {res: [] for res in self.resolutions}

        self.model = model
        self.prompt_csv = prompt_csv
        self.qps_csv = qps_csv
        self.init_time = init_time

        self.index = 0
        self.max = 500

        self.world_size = world_size

    def set_init_time(self, init_time):
        self.init_time = init_time

    def add_req(self, req: Request):
        self.pool[req.resolution].append(req)

    def get_oldest_res(self):
        index = -1
        for i in range(len(self.resolutions)):
            if self.pool[self.resolutions[i]]:
                index = i
                break
        if index == -1:
            return None

        target_res = self.resolutions[index]

        for i in range(len(self.resolutions)):
            res = self.resolutions[i]
            if self.pool[res]:
                if self.pool[res][0].delay_time < self.pool[target_res][0].delay_time:
                    target_res = res
        return target_res

    def has_unfinished_reqs(self):
        for res, l in self.pool.items():
            if len(l) > 0:
                return True
        return False

    def get_next_reqs(self, timepoint) -> Tuple[int, List[Request]]:
        if self.index >= self.max and not self.has_unfinished_reqs():
            print(f"{self.index=}, {self.max=}")
            return None, None

        # Add until newest request
        i = self.index
        while self.index < self.max:
            delay_time = self.qps_csv.iloc[i, 0]
            if (timepoint - self.init_time) * 1000 < delay_time:
                break
            prompt = self.prompt_csv.iloc[i, 1]
            res = self.qps_csv.iloc[i, 1]
            step = self.qps_csv.iloc[i, 2]

            self.add_req(
                Request(
                    index=i,
                    resolution=res,
                    step=step,
                    prompt=prompt,
                    delay_time=delay_time,
                )
            )

            i += 1
            if i >= self.max:
                break
        self.index = i

        oldest_res = self.get_oldest_res()
        if oldest_res is None:
            return -1, []
        else:
            max_bs = setup[self.model]["max_bs"][self.world_size][oldest_res]
            if max_bs > 12:
                max_bs = 12
            old_req_list = self.pool[oldest_res]
            ret = old_req_list[:max_bs]
            self.pool[oldest_res] = old_req_list[max_bs:]
            return oldest_res, ret
