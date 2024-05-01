from typing import List, Tuple, Dict, Union, Optional, TYPE_CHECKING

import torch
import numpy as np

from diffusers import EulerDiscreteScheduler as DiffusersEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from .utils import BatchSupportScheduler, BaseSchedulerStates

if TYPE_CHECKING:
    from sduss.worker import WorkerRequest

class EulerDiscreteSchedulerStates(BaseSchedulerStates):
    """Scheduler states wrapper to store scheduler states of each request."""
    total_steps_dependent_attr_names = [
        "sigmas",
        "num_inference_steps",
        "timesteps",
    ]
    current_step_dependent_attr_names = [
        "_step_index"
    ]
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.sigmas = kwargs.pop("sigmas")
        self.num_inference_steps = kwargs.pop("num_inference_steps")
        self.timesteps = kwargs.pop("timesteps")

        # self._step_index = kwargs.pop("_step_index")
        self._step_index = 0

    
    def update_states_one_step(self):
        self.timestep_idx += 1
        # self._step_index += 1  # Handled in `step` method
        assert self.timestep_idx <= self.timesteps.shape[0]

    
    def get_next_timestep(self):
        return self.timesteps[self.timestep_idx]

    
    def get_step_idx(self):
        assert self.timestep_idx == self._step_index
        return self.timestep_idx


class EulerDiscreteScheduler(DiffusersEulerDiscreteScheduler, BatchSupportScheduler):
    def batch_set_timesteps(
        self,
        worker_reqs: List["WorkerRequest"],
        device: torch.device,
    ):
        """Set timesteps method with batch support

        Args:
            worker_reqs (List[WorkerRequest]): Requests to set timesteps
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
            for name in EulerDiscreteSchedulerStates.total_steps_dependent_attr_names:
                attrs[name] = getattr(self, name)
            for name in EulerDiscreteSchedulerStates.current_step_dependent_attr_names:
                attrs[name] = getattr(self, name)
            for req in collected_reqs:
                req.scheduler_states = EulerDiscreteSchedulerStates(**attrs)

            # 5. set step_index here, so that we don't need to call it at the first time
            # scale_model_output is called
            self._batch_init_step_index(collected_reqs)
            

    def _batch_index_for_timestep(
        self, 
        req: "WorkerRequest",
    ):
        """Batch compatible version method.
        
        Reqs with same total inference steps will have the same index.
        The return value can be broadcast.

        Args:
            timestep (_type_): _description_
        """
        # TODO(MX): We may return 1 directly here.
        # schedule_timesteps = req.scheduler_states.timesteps
        # indices = (schedule_timesteps == timestep).nonzero()
        # pos = 1 if len(indices) > 1 else 0
        # return indices[pos].item()
        
        # if (req.scheduler_states.timesteps.shape[0] > 1):
        #     return 1
        # else:
        #     return 0
    
        return 0
        

    def _batch_init_step_index(
        self, 
        worker_reqs_with_same_total_steps: List["WorkerRequest"],
    ):
        """Batch compatible version method.

        Since reqs with the same total steps will all call this method
        at the prepare stage, they will share the same _step_index. 
        We can broadcast it among the reqs.
        This method must be called at the prepare stage!

        Args:
            worker_reqs_with_same_total_steps (List[WorkerRequest]): _description_
            timestep (Union[float, torch.FloatTensor]): _description_
        """
        example_req = worker_reqs_with_same_total_steps[0]
        _step_index = self._batch_index_for_timestep(example_req)
        for req in worker_reqs_with_same_total_steps:
            req.scheduler_states._step_index = _step_index
        
    
    def batch_scale_model_input(
        self,
        worker_reqs: List["WorkerRequest"],
        samples: torch.Tensor,
        timestep_list: torch.Tensor,
    ) -> torch.Tensor:
        # Since step_index is initialized in set_timestep, we don't need
        # to do it again here, just check it
                
        # Collect sigmas
        collected_sigmas = []
        for req in worker_reqs:
            req_sigma = req.scheduler_states.sigmas[req.scheduler_states._step_index]
            collected_sigmas.append(req_sigma)
        sigmas_torch = torch.tensor(data=collected_sigmas, dtype=samples.dtype).to(samples.device)
        shape = [samples.shape[0]] + [1] *(samples.ndim - 1)
        if shape[0] == sigmas_torch.shape[0] * 2:
            # classifier free
            sigmas_torch = sigmas_torch.repeat(2)
        sigmas_torch = sigmas_torch.reshape(shape=shape)

        samples = samples / ((sigmas_torch ** 2 + 1) ** 0.5)

        return samples
    
    
    def batch_step(
        self,
        worker_reqs: List["WorkerRequest"],
        model_outputs: torch.Tensor,
        timestep_list: torch.Tensor,
        samples: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:

        # These parameters are fixed, so that we can simplify the procedure
        if (s_churn != 0.0 or 
            s_tmin != 0.0 or 
            s_tmax != float("inf") or
            s_noise != 1.0 or
            generator is not None):
            raise NotImplementedError("We do not support custom parameters at this time.")

        # 1. Upcast to avoid precision issues
        samples = samples.to(torch.float32)

        # 2. collect sigma and calculate as a batch
        collected_sigmas = []
        for req in worker_reqs:
            req_sigma = req.scheduler_states.sigmas[req.scheduler_states._step_index]
            collected_sigmas.append(req_sigma)
        sigmas_torch = torch.tensor(data=collected_sigmas, dtype=torch.float32).to(torch.cuda.current_device())
        shape = [len(collected_sigmas)] + [1] *(model_outputs.ndim - 1)
        sigmas_torch = sigmas_torch.reshape(shape=shape)
        
        # 3. Calculate Gamma
        # Since we've set the defualt parameters, gammas must be 0
        gamma = torch.zeros_like(sigmas_torch)
        
        # 4. create noise
        noise = randn_tensor(model_outputs.shape, dtype=model_outputs.dtype, device=model_outputs.device, generator=None)
        
        # We have s_noise = 1
        esp = noise
        sigma_hat_torch = sigmas_torch
        
        # Gamma must be 0
        # if gamma > 0:
        #     sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 5. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_outputs
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = samples - sigma_hat_torch * model_outputs
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_outputs * (-sigmas_torch / (sigmas_torch**2 + 1) ** 0.5) + (samples / (sigmas_torch**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 6. Convert to an ODE derivative
        derivative = (samples - pred_original_sample) / sigma_hat_torch

        # Colelct next sigmas
        collected_next_sigmas = []
        for req in worker_reqs:
            req_sigma = req.scheduler_states.sigmas[req.scheduler_states._step_index + 1]
            collected_next_sigmas.append(req_sigma)
        next_sigmas_torch = torch.tensor(data=collected_next_sigmas, dtype=sigmas_torch.dtype).to(
                            sigmas_torch.device).reshape(sigmas_torch.shape)
        dt = next_sigmas_torch - sigma_hat_torch
        # dt = self.sigmas[self.step_index + 1] - sigma_hat_torch

        prev_sample = samples + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_outputs.dtype)

        # upon completion increase step index by one
        # self._step_index += 1
        for req in worker_reqs:
            req.scheduler_states._step_index += 1

        return prev_sample