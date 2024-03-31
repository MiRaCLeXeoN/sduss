from typing import List, Iterable

import numpy as np
import torch

from diffusers import PNDMScheduler as DiffusersPNDMScheduler

from .utils import BatchSupportScheduler

from sduss.worker import WorkerRequest

class PNDMSchedulerStates:
    total_steps_dependent_attr_names = [
        "prk_timesteps",
        "num_inference_steps",
        ""
    ]
    current_step_dependent_attr_names = [
        
    ]
    """Scheduler states wrapper to store scheduler states of each request."""
    def __init__(self) -> None:
        pass


class PNDMSCheduler(DiffusersPNDMScheduler, BatchSupportScheduler):
    def batch_set_timesteps(
        self,
        worker_reqs: List[WorkerRequest],
        device: torch,
    ) -> None:
        """Set timesteps method with batch support

        Args:
            worker_reqs (List[WorkerRequest]): Requests to set timesteps
        """
        # 1. sort the reqs so according to num_inference_steps
        sorted_worker_reqs = sorted(worker_reqs, key=lambda req: req.sampling_params.num_inference_steps, reverse=False)
        total_reqs_count = len(worker_reqs)
        
        for i in range(len(worker_reqs)):
            # 2. group reqs that have same inference steps, so that we can
            # copy data among them
            collected_reqs = [worker_reqs[i]]
            i += 1
            target_num_inference_steps = collected_reqs[0].sampling_params.num_inference_steps
            while i < total_reqs_count and worker_reqs[i].sampling_params.num_inference_steps == target_num_inference_steps:
                collected_reqs.append(worker_reqs[i])
                i += 1
            
            # 3. Do the basic version of set_timesteps
            self.set_timesteps(num_inference_steps=target_num_inference_steps, device=device)
            
            # 4. Extract the necessary results from `self` and store them in
            # wrappers in each reqs
        
        
        
        if self.config.timestep_spacing == "linspace":
            self._timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps).round().astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            self._timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
            self._timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            self._timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio))[::-1].astype(
                np.int64
            )
            self._timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
        

        
        
        return super().batch_set_timesteps()


    def batch_scale_model_input(
        self,
        worker_reqs: List[WorkerRequest],
    ):
        """Nothing to do. PNDM doesn't need to scale model input."""
        return 
    
    
    def batch_step(self):
        return super().batch_step()

