from typing import List, Iterable

import numpy as np
import torch

from diffusers import PNDMScheduler as DiffusersPNDMScheduler

from .utils import BatchSupportScheduler, BaseSchedulerStates

from sduss.worker import WorkerRequest

class PNDMSchedulerStates(BaseSchedulerStates):
    total_steps_dependent_attr_names = [
        "prk_timesteps",
        "num_inference_steps",
        "timesteps",
    ]
    current_step_dependent_attr_names = [
        "counter",
        "cur_model_output",
        "ets",
        "cur_sample",
    ]
    """Scheduler states wrapper to store scheduler states of each request."""
    def __init__(self, **kwargs) -> None:
        self.prk_timesteps: np.ndarray = kwargs.pop("prk_timesteps")
        self.num_inference_steps: int = kwargs.pop("num_inference_steps")
        self.timesteps: torch.FloatTensor = kwargs.pop("timesteps")

        self.counter: int = kwargs.pop("counter")
        self.cur_model_output: torch.FloatTensor = kwargs.pop("cur_model_output")
        self.ets: List[torch.FloatTensor] = kwargs.pop("ets")
        self.cur_sample: torch.FloatTensor = kwargs.pop("cur_sample")

        # Common variables
        super().__init__()
    
    def update_staets_one_step(self):
        self.timestep_idx += 1
        assert self.timestep_idx <= self.timesteps.shape[0]
    
    def get_next_timestep(self):
        return self.timesteps[self.timestep_idx]


class PNDMSCheduler(DiffusersPNDMScheduler, BatchSupportScheduler):
    def batch_set_timesteps(
        self,
        worker_reqs: List[WorkerRequest],
        device: torch.device,
    ) -> None:
        """Set timesteps method with batch support

        Args:
            worker_reqs (List[WorkerRequest]): Requests to set timesteps
        """
        # 1. sort the reqs according to num_inference_steps
        worker_reqs = sorted(worker_reqs, key=lambda req: req.sampling_params.num_inference_steps, reverse=False)
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
            attrs = {}
            for name in PNDMSchedulerStates.total_steps_dependent_attr_names:
                attrs[name] = getattr(self, name)
            for name in PNDMSchedulerStates.current_step_dependent_attr_names:
                attrs[name] = getattr(self, name)
            for req in collected_reqs:
                req.scheduler_states = PNDMSchedulerStates(**attrs)


    def batch_scale_model_input(
        self,
        worker_reqs: List[WorkerRequest],
    ):
        """Nothing to do. PNDM doesn't need to scale model input."""
        return 
    
    
    def batch_step(
        self,
        worker_reqs: List[WorkerRequest],
        model_outputs: List[torch.FloatTensor],
        timestep_list: List[int],
    ):
        """Batch-compatible step method.

        WorkerRequest's latent must be updated before this method call.

        Args:
            worker_reqs (List[WorkerRequest]): _description_
            model_outputs (List[torch.FloatTensor]): _description_
            timestep_list (List[int]): _description_

        Returns:
            _type_: _description_
        """
        # 1. Split requests according to their counters. Decide which ones should go
        # prk and which ones go plms
        prk_reqs = []
        prk_model_outputs = []
        prk_timestep_list = []
        plms_reqs = []
        plms_model_outputs = []
        plms_timestep_list = []
        for i, req in enumerate(worker_reqs):
            if req.scheduler_states.counter < len(req.scheduler_states.prk_timesteps) and self.config.skip_prk_steps:
                prk_reqs.append(req)
                prk_model_outputs.append(model_outputs[i])
                prk_timestep_list.append(timestep_list[i])
            else:
                plms_reqs.append(req)
                plms_model_outputs.append(model_outputs[i])
                plms_timestep_list.append(timestep_list[i])
        
        self.batch_step_prk(prk_reqs, prk_model_outputs, prk_timestep_list)
        self.batch_step_plms(plms_reqs, plms_model_outputs, plms_timestep_list)

        return super().batch_step()
    
    
    def batch_step_prk(
        self,
        worker_reqs: List[WorkerRequest],
        model_outputs: List[torch.FloatTensor],
        timestep_list: List[int],
    ):
        """Batch-compatible step prk method.
        
        This method functions sequentially in nature.

        Args:
            worker_reqs (List[WorkerRequest]): _description_
            model_outputs (List[torch.FloatTensor]): _description_
            timestep_list (List[int]): _description_
        """
        ret = []
        for i, req in enumerate(worker_reqs):
            model_output = model_outputs[i]
            timestep = timestep_list[i]
            sample = req.sampling_params.latents

            diff_to_prev = 0 if req.scheduler_states.counter % 2 else (
                req.scheduler_states.config.num_train_timesteps // req.scheduler_states.num_inference_steps // 2)
            prev_timestep = timestep - diff_to_prev
            timestep = req.scheduler_states.prk_timesteps[req.scheduler_states.counter // 4 * 4]

            if req.scheduler_states.counter % 4 == 0:
                req.scheduler_states.cur_model_output += 1 / 6 * model_output
                req.scheduler_states.ets.append(model_output)
                req.scheduler_states.cur_sample = sample
            elif (req.scheduler_states.counter - 1) % 4 == 0:
                req.scheduler_states.cur_model_output += 1 / 3 * model_output
            elif (req.scheduler_states.counter - 2) % 4 == 0:
                req.scheduler_states.cur_model_output += 1 / 3 * model_output
            elif (req.scheduler_states.counter - 3) % 4 == 0:
                model_output = req.scheduler_states.cur_model_output + 1 / 6 * model_output
                req.scheduler_states.cur_model_output = 0

            # cur_sample should not be `None`
            cur_sample = req.scheduler_states.cur_sample if req.scheduler_states.cur_sample is not None else sample

            prev_sample = self._get_prev_sample(cur_sample, timestep, prev_timestep, model_output)
            req.scheduler_states.counter += 1
            
            ret.append((prev_sample, ))
        return ret

    
    def batch_step_plms(
        self,
        worker_reqs: List[WorkerRequest],
        model_outputs: List[torch.FloatTensor],
        timestep_list: List[int],
    ):
        ret = []
        for i, req in enumerate(worker_reqs):
            model_output = model_outputs[i]
            timestep = timestep_list[i]
            sample = req.sampling_params.latents

            prev_timestep = timestep - self.config.num_train_timesteps // req.scheduler_states.num_inference_steps

            if req.scheduler_states.counter != 1:
                req.scheduler_states.ets = req.scheduler_states.ets[-3:]
                req.scheduler_states.ets.append(model_output)
            else:
                prev_timestep = timestep
                timestep = timestep + req.scheduler_states.config.num_train_timesteps // req.scheduler_states.num_inference_steps

            if len(req.scheduler_states.ets) == 1 and req.scheduler_states.counter == 0:
                model_output = model_output
                req.scheduler_states.cur_sample = sample
            elif len(req.scheduler_states.ets) == 1 and req.scheduler_states.counter == 1:
                model_output = (model_output + req.scheduler_states.ets[-1]) / 2
                sample = req.scheduler_states.cur_sample
                req.scheduler_states.cur_sample = None
            elif len(req.scheduler_states.ets) == 2:
                model_output = (3 * req.scheduler_states.ets[-1] - req.scheduler_states.ets[-2]) / 2
            elif len(req.scheduler_states.ets) == 3:
                model_output = (23 * req.scheduler_states.ets[-1] - 16 * req.scheduler_states.ets[-2] + 5 * req.scheduler_states.ets[-3]) / 12
            else:
                model_output = (1 / 24) * (55 * req.scheduler_states.ets[-1] - 59 * req.scheduler_states.ets[-2] + 37 * req.scheduler_states.ets[-3] - 9 * req.scheduler_states.ets[-4])

            prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, model_output)
            req.scheduler_states.counter += 1

            ret.append((prev_sample, ))
        
        return ret

