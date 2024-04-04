from typing import List, Tuple, Dict, Union, Optional

import torch
import numpy as np

from diffusers import EulerDiscreteScheduler as DiffusersEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from .utils import BatchSupportScheduler
from sduss.worker import WorkerRequest
from sduss.scheduler import SUPPORT_RESOLUTION

class EulerDiscreteSchedulerStates:
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
        self.sigmas = kwargs.pop("sigmas")
        self.num_inference_steps = kwargs.pop("num_inference_steps")
        self.timesteps = kwargs.pop("timesteps")

        self._step_index = kwargs.pop("_step_index")

class EulerDiscreteScheduler(DiffusersEulerDiscreteScheduler, BatchSupportScheduler):
    def batch_set_timesteps(
        self,
        worker_reqs: List[WorkerRequest],
        device: torch.device,
    ):
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
        req: WorkerRequest,
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
        
        if (req.scheduler_states.timestpes.shape[0] > 1):
            return 1
        else:
            return 0
        

    def _batch_init_step_index(
        self, 
        worker_reqs_with_same_total_steps: List[WorkerRequest],
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
        worker_reqs: List[WorkerRequest],
        timestep_list: List[Union[float, torch.FloatTensor]],
    ):
        # Since step_index is initialized in set_timestep, we don't need
        # to do it again here, just check it
                
        # 2. Scale group by group
        worker_reqs = sorted(worker_reqs, key=lambda req: req.sampling_params.num_inference_steps, reverse=False)
        total_reqs_count = len(worker_reqs)
        
        for i in range(len(worker_reqs)):
            # group reqs that have same inference steps, so that we can copy data among them
            collected_reqs = [worker_reqs[i]]
            i += 1
            target_num_inference_steps = collected_reqs[0].sampling_params.num_inference_steps
            while i < total_reqs_count and worker_reqs[i].sampling_params.num_inference_steps == target_num_inference_steps:
                collected_reqs.append(worker_reqs[i])
                i += 1
            
            sigmas = collected_reqs[0].scheduler_states.sigmas
            for req in collected_reqs:
                step_index = req.scheduler_states._step_index
                sigma = sigmas[step_index]
                req.sampling_params.latents = req.sampling_params.latents / ((sigma**2 + 1) ** 0.5)
        
        return None
    
    def batch_step(
        self,
        worker_reqs: List[WorkerRequest],
        model_outputs: List[torch.FloatTensor],
        timestep_list: List[Union[float, torch.FloatTensor]],
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> List[Tuple]:

        # These parameters are fixed, so that we can simplify the procedure
        if (s_churn != 0.0 or 
            s_tmin != 0.0 or 
            s_tmax != float("inf") or
            s_noise != 0.0 or
            generator is not None):
            raise NotImplementedError("We do not support custom parameters at this time.")

        # 1. Group reqs by resolution
        worker_reqs_dict: Dict[int, List[WorkerRequest]] = {res:[] for res in SUPPORT_RESOLUTION}
        for req in worker_reqs:
            req_resolution = req.sampling_params.height
            worker_reqs_dict[req_resolution].append(req)
        
        # 2. collect sigma and calculate as a batch
        collected_sigmas = []
        req_list_start_idx = []
        i = 0
        for req_list in worker_reqs_dict.values():
            req_list_start_idx.append(i)
            for req in req_list:
                req_sigma = req.scheduler_states.sigmas[req.scheduler_states._step_index]
                collected_sigmas.append(req_sigma)
                i += 1
        
        sigmas_np = np.array(collected_sigmas, dtype=np.float32)
        sigmas_torch = torch.from_numpy(sigmas_np).to(torch.cuda.current_device())
        # split sigmas to each resolution
        sigma_dict: Dict[int, torch.Tensor] = {}
        for resolution, chunk in map(worker_reqs_dict.keys(), 
                                     sigmas_torch.tensor_split(req_list_start_idx[1:])):
            sigma_dict[resolution] = chunk
        
        # 3. Calculate Gamma
        # Since we've set the defualt parameters, gammas must be 0
        # gamma = torch.zeros_like(sigmas_torch)
        
        # 4. Process each resolution
        for res, reqs in worker_reqs_dict.items():
            samples = [req.sampling_params.latents.unsqueeze() for req in reqs]
            cat_samples = torch.cat(samples, dim=0)

            # 3. create noise
            example_latent = reqs[0].sampling_params.latents
            shape = [len(reqs), *example_latent.shape]

            noise = randn_tensor(shape, dtype=example_latent.dtype, device=example_latent.device)
            
            # s_noise = 1
            esp = noise
            sigma_hat = sigma_dict[res]
            
            if self.config.prediction_type == "epsilon":
                pred_original_sample = 
                
            
            
                

            
    
        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # sigma = self.sigmas[self.step_index]

        # gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        # noise = randn_tensor(
        #     model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        # )

        # eps = noise * s_noise
        # sigma_hat = sigma * (gamma + 1)

        # if gamma > 0:
        #     sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        
        
        return None