import argparse
import csv
import os
import random
import time
import datetime
import gc

import torch
import torch.distributed as dist

from utils import get_pipeline, get_csv_files, RequestPool

if __name__ == "__main__":

    slo = os.environ.get("SLO", "5")
    qps = os.environ.get("QPS", "8.0")
    model = os.environ.get("MODEL", "sdxl")
    num = int(os.environ.get("NUM", "500"))

    # Get files
    distri_config, pipeline = get_pipeline(model)
    prompt_csv, qps_csv = get_csv_files(model, qps)

    rank = distri_config.rank
    print(f"rank {rank} start")
    # Prepare log
    if rank == 0:
        res_dir_path = f"./results/{model}/{qps}_{slo}_distrifusion_{dist.get_world_size()}"
        os.makedirs(res_dir_path, exist_ok=True)

        client_f = open(os.path.join(res_dir_path, "client.csv"), "w")
        fields = ["index", "success", "request_id", "resolution", "step", "delay_time", "arrival_time", "start_time", "end_time", "time_consumption", "prompt"]
        client_csv = csv.DictWriter(client_f, fieldnames=fields)
        client_csv.writeheader()
    
    world_size = dist.get_world_size()
    gloo_pg = dist.new_group(backend="gloo", ranks=list(range(world_size)))

    if rank == 0:
        init_time = time.time()
        init_datetime = datetime.datetime.now()
    else:
        init_time = None
        init_datetime = None
    objs = [init_time, init_datetime]
    dist.broadcast_object_list(objs, src=0, group=gloo_pg)
    init_time, init_datetime = objs
    request_pool = RequestPool(model, prompt_csv, qps_csv, init_time, world_size=world_size, max=num)

    print(f"{rank=} ready, {init_time=}, {init_datetime=}")
    random.seed(10086)
    generator = torch.Generator(device="cuda").manual_seed(10086)

    cpu_device = torch.device("cpu")
    while True:
        if rank == 0:
            timepoint = time.time()
            res, reqs = request_pool.get_next_reqs(timepoint)
            if res == -1:
                # print(f"no valid requests found, continue")
                continue
        else:
            res, reqs = None, None
        objs = [res, reqs]
        dist.broadcast_object_list(objs, src=0, group=gloo_pg)
        res, reqs = objs

        if res is None:
            print(f"exit")
            break

        gc.collect()
        torch.cuda.empty_cache()

        # Prepare requests
        prompts = []
        for req in reqs:
            prompts.append(req.prompt)
        
        if rank == 0:
            req_ids = [req.index for req in reqs]
            print(f"{len(reqs)=}, {res=}, reqs={req_ids}")
        
        start_time = datetime.datetime.now()
        pipeline(
            height=res,
            width=res,
            prompt=prompts,
            num_inference_steps=50,
            generator=generator,
        )
        end_time = datetime.datetime.now()

        if rank == 0:
            print(f"before:\nmemory allocated: {torch.cuda.memory_allocated() / (1024**3)}, \n"
                  f"memory reserved: {torch.cuda.memory_reserved() / (1024**3)}, \n"
                  f"max_memory_allocated: {torch.cuda.max_memory_allocated() / (1024**3)}")

        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"after:\nmemory allocated: {torch.cuda.memory_allocated() / (1024**3)}, \n"
                  f"memory reserved: {torch.cuda.memory_reserved() / (1024**3)}, \n"
                  f"max_memory_allocated: {torch.cuda.max_memory_allocated() / (1024**3)}")

        if rank == 0:
            for req in reqs:
                # print(f"finish req {req.index}")
                arrival_time = init_datetime + datetime.timedelta(milliseconds=req.delay_time)
                client_csv.writerow({
                    "index" : req.index, 
                    "success" : True, 
                    "request_id" : req.request_id, 
                    "resolution" : req.resolution, 
                    "step" : 50, 
                    "delay_time" : req.delay_time, 
                    "arrival_time" : arrival_time,
                    "start_time" : start_time,
                    "end_time" : end_time, 
                    "time_consumption" : (end_time - arrival_time).total_seconds(), 
                    "prompt" : req.prompt,
                })

