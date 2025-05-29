from typing import List, Iterable, Optional, Union, Dict, TYPE_CHECKING

import numpy as np
import torch

from diffusers import PNDMScheduler as DiffusersPNDMScheduler

from sduss.logger import init_logger

from .utils import BatchSupportScheduler, BaseSchedulerStates

if TYPE_CHECKING:
    from sduss.worker.runner.wrappers import RunnerRequest

logger = init_logger(__name__)

class PNDMSchedulerStates(BaseSchedulerStates):
    """Scheduler states wrapper to store scheduler states of each request."""
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
    def __init__(self, **kwargs) -> None:
        # Common variables
        super().__init__()

        self.prk_timesteps: np.ndarray = kwargs.pop("prk_timesteps")
        self.num_inference_steps: int = kwargs.pop("num_inference_steps")
        self.timesteps: torch.FloatTensor = kwargs.pop("timesteps")

        self.counter: int = kwargs.pop("counter")
        self.cur_model_output: torch.FloatTensor = kwargs.pop("cur_model_output")
        self.ets: List[torch.FloatTensor] = kwargs.pop("ets")
        self.cur_sample: torch.FloatTensor = kwargs.pop("cur_sample")

    
    def update_states_one_step(self):
        self.timestep_idx += 1
        assert self.timestep_idx <= self.timesteps.shape[0]
    
    def get_next_timestep(self):
        return self.timesteps[self.timestep_idx]

    
    def get_step_idx(self):
        return self.timestep_idx

    
    def log_status(self):
        logger.debug(f"{self.num_inference_steps=}, {self.counter=}, {len(self.ets)=}, {self.timestep_idx=}")
        logger.debug(f"{self.prk_timesteps.shape=}, current_time_step={self.timesteps[self.timestep_idx]}")
    
    
    def to_device(self, device) -> None:
        # counter
        # cur_model_output
        # ets
        # cur_sample
        # num_inference_steps
        # prk_timesteps: np.ndarray
        self.timesteps = self.timesteps.to(device=device)
    
    
    def to_dtype(self, dtype) -> None:
        pass

    
    def to_numpy(self) -> None:
        self.timesteps = self.timesteps.numpy()
    
    
    def to_tensor(self) -> None:
        self.timesteps = torch.from_numpy(self.timesteps)


class PNDMScheduler(DiffusersPNDMScheduler, BatchSupportScheduler):
    def batch_set_timesteps(
        self,
        worker_reqs: List["RunnerRequest"],
        device: torch.device,
    ) -> None:
        """Set timesteps method with batch support.
        
        set_timesteps don't need to take image size into consideration.

        Args:
            worker_reqs (List[RunnerRequest]): Requests to set timesteps
        """
        # 1. sort the reqs according to num_inference_steps
        worker_reqs = sorted(worker_reqs, key=lambda req: req.sampling_params.num_inference_steps, reverse=False)
        total_reqs_count = len(worker_reqs)
        
        i = 0
        while i < total_reqs_count:
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
        worker_reqs: List["RunnerRequest"],
        samples: torch.Tensor,
        timestep_list: torch.Tensor,
    ):
        """Batch support scale model input.

        PNDM does nothing here. 

        Args:
            latent_model_input (torch.FloatTensor): Latent model input is a batched input.
            timesteps (torch.FloatTensor): Timesteps should have the same batchsize.
        """
        return samples
    
    
    def batch_step(
        self,
        worker_reqs: List["RunnerRequest"],
        model_outputs: torch.FloatTensor,
        timestep_list: torch.Tensor,
        samples: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.FloatTensor:
        """Batch-compatible step method.

        RunnerRequest's latent must be updated before this method call.

        Args:
            worker_reqs (List[RunnerRequest]): _description_
            model_outputs (List[torch.FloatTensor]): _description_
            timestep_list (List[int]): _description_

        Returns:
            torch.Tensor: concatenated results.
        """
        # 1. Split requests according to their counters. Decide which ones should go
        # prk and which ones go plms
        prk_reqs = []
        prk_model_outputs = []
        prk_timestep_list = []
        prk_sample_list = []
        prk_idx = []
        plms_reqs = []
        plms_model_outputs = []
        plms_timestep_list = []
        plms_sample_list = []
        plms_idx = []
        for i, req in enumerate(worker_reqs):
            if req.scheduler_states.counter < len(req.scheduler_states.prk_timesteps) and self.config.skip_prk_steps:
                prk_reqs.append(req)
                prk_model_outputs.append(model_outputs[i])
                prk_timestep_list.append(timestep_list[i])
                prk_sample_list.append(samples[i])
                prk_idx.append(i)
            else:
                plms_reqs.append(req)
                plms_model_outputs.append(model_outputs[i])
                plms_timestep_list.append(timestep_list[i])
                plms_sample_list.append(samples[i])
                plms_idx.append(i)
        
        prk_ret = self.batch_step_prk(prk_reqs, prk_model_outputs, prk_timestep_list, prk_timestep_list)
        plms_ret = self.batch_step_plms(plms_reqs, plms_model_outputs, plms_sample_list, plms_timestep_list)

        # sort to original order
        ret: List[torch.Tensor] = []
        prki = 0
        plmsi = 0
        while prki < len(prk_idx) and plmsi < len(plms_idx):
            if prk_idx[prki] < plms_idx[plmsi]:
                # this req is from prk
                ret.append(prk_ret[prki])
                prki += 1
            else:
                ret.append(plms_idx[plms_idx])
                plmsi += 1
        while prki < len(prk_idx):
            ret.append(prk_ret[prki])
            prki += 1
        while plmsi < len(plms_idx):
            ret.append(plms_ret[plmsi])
            plmsi += 1
        
        ret = [t.unsqueeze(dim=0) for t in ret]
        ret = torch.cat(ret, dim=0)
        return ret

   
    def batch_step_prk(
        self,
        worker_reqs: List["RunnerRequest"],
        model_outputs: List[torch.FloatTensor],
        samples: List[torch.FloatTensor],
        timestep_list: List[int],
    ):
        """Batch-compatible step prk method.
        
        This method functions sequentially in nature, since requests
        are likely to diverge and it's hard to batch.

        Args:
            worker_reqs (List[RunnerRequest]): _description_
            model_outputs (List[torch.FloatTensor]): _description_
            timestep_list (List[int]): _description_
        """
        ret = []
        for i, req in enumerate(worker_reqs):
            model_output = model_outputs[i]
            timestep = timestep_list[i]
            sample = samples[i]

            diff_to_prev = 0 if req.scheduler_states.counter % 2 else (
                self.config.num_train_timesteps // req.scheduler_states.num_inference_steps // 2)
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
            
            ret.append(prev_sample)
        return ret

    
    def batch_step_plms(
        self,
        worker_reqs: List["RunnerRequest"],
        model_outputs: List[torch.FloatTensor],
        samples: List[torch.FloatTensor],
        timestep_list: List[int],
    ):
        """Batch support version.

        This method functions sequentially in nature, since requests
        are likely to diverge and it's hard to batch.

        Args:
            worker_reqs (List[RunnerRequest]): _description_
            model_outputs (List[torch.FloatTensor]): _description_
            timestep_list (List[int]): _description_
        """
        ret = []
        for i, req in enumerate(worker_reqs):
            model_output = model_outputs[i]
            timestep = timestep_list[i]
            sample = samples[i]

            prev_timestep = timestep - self.config.num_train_timesteps // req.scheduler_states.num_inference_steps

            if req.scheduler_states.counter != 1:
                req.scheduler_states.ets = req.scheduler_states.ets[-3:]
                req.scheduler_states.ets.append(model_output)
            else:
                prev_timestep = timestep
                timestep = timestep + self.config.num_train_timesteps // req.scheduler_states.num_inference_steps

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

            ret.append(prev_sample)
        
        return ret

