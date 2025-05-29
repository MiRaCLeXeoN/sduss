from typing import List, Tuple, Dict, Union, Optional, TYPE_CHECKING

import torch

from diffusers import FlowMatchEulerDiscreteScheduler as DiffusersFlowMatchEulerDiscreteScheduler

from .utils import BatchSupportScheduler, BaseSchedulerStates

if TYPE_CHECKING:
    from sduss.worker.runner.wrappers import RunnerRequest

class FlowMatchEulerDiscreteSchedulerStates(BaseSchedulerStates):
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
    
    
    def to_device(self, device) -> None:
        # self.sigmas should be on CPU
        self.sigmas = self.sigmas.to("cpu")
        self.timesteps = self.timesteps.to(device=device)
    

    def to_dtype(self, dtype) -> None:
        # We should not alter any dtype
        pass

    
    def to_numpy(self) -> None:
        self.sigmas = self.sigmas.numpy()
        self.timesteps = self.timesteps.numpy()
    
    
    def to_tensor(self) -> None:
        self.sigmas = torch.from_numpy(self.sigmas)
        self.timesteps = torch.from_numpy(self.timesteps)


class FlowMatchEulerDiscreteScheduler(DiffusersFlowMatchEulerDiscreteScheduler, BatchSupportScheduler):
    def batch_set_timesteps(
        self,
        runner_reqs: List["RunnerRequest"],
        device: torch.device,
    ):
        """Set timesteps method with batch support

        Args:
            runner_reqs (List[RunnerRequest]): Requests to set timesteps
        """
        # 1. sort the reqs according to num_inference_steps
        runner_reqs = sorted(runner_reqs, key=lambda req: req.sampling_params.num_inference_steps, reverse=False)
        total_reqs_count = len(runner_reqs)
        
        i = 0
        while i < total_reqs_count:
            # 2. group reqs that have same inference steps, so that we can
            # copy data among them
            collected_reqs = [runner_reqs[i]]
            i += 1
            target_num_inference_steps = collected_reqs[0].sampling_params.num_inference_steps
            while i < total_reqs_count and runner_reqs[i].sampling_params.num_inference_steps == target_num_inference_steps:
                collected_reqs.append(runner_reqs[i])
                i += 1
            
            # 3. Do the basic version of set_timesteps
            self.set_timesteps(num_inference_steps=target_num_inference_steps, device=device)

            # 4. Extract the necessary results from `self` and store them in
            # wrappers in each reqs
            attrs = {}
            for name in FlowMatchEulerDiscreteSchedulerStates.total_steps_dependent_attr_names:
                attrs[name] = getattr(self, name)
            for name in FlowMatchEulerDiscreteSchedulerStates.current_step_dependent_attr_names:
                attrs[name] = getattr(self, name)
            for req in collected_reqs:
                req.scheduler_states = FlowMatchEulerDiscreteSchedulerStates(**attrs)

            # 5. set step_index here, so that we don't need to call it at the first time
            # scale_model_output is called
            self._batch_init_step_index(collected_reqs)
            

    def _batch_index_for_timestep(
        self, 
        req: "RunnerRequest",
    ):
        """Batch compatible version method.
        
        Reqs with same total inference steps will have the same index.
        The return value can be broadcast.

        Args:
            timestep (_type_): _description_
        """
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
        runner_reqs_with_same_total_steps: List["RunnerRequest"],
    ):
        """Batch compatible version method.

        Since reqs with the same total steps will all call this method
        at the prepare stage, they will share the same _step_index. 
        We can broadcast it among the reqs.
        This method must be called at the prepare stage!

        Args:
            runner_reqs_with_same_total_steps (List[RunnerRequest]): _description_
            timestep (Union[float, torch.FloatTensor]): _description_
        """
        example_req = runner_reqs_with_same_total_steps[0]
        _step_index = self._batch_index_for_timestep(example_req)
        for req in runner_reqs_with_same_total_steps:
            req.scheduler_states._step_index = _step_index
        
    
    def batch_step(
        self,
        runner_reqs: List["RunnerRequest"],
        model_outputs: torch.FloatTensor,
        samples: torch.FloatTensor,
        timesteps: Union[float, torch.FloatTensor],
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:

        # 1. Upcast to avoid precision issues
        samples = samples.to(torch.float32)

        # 2. collect sigma and calculate as a batch
        collected_sigmas = []
        collected_sigmas_next = []
        for req in runner_reqs:
            req_sigma = req.scheduler_states.sigmas[req.scheduler_states._step_index]
            req_sigma_next = req.scheduler_states.sigmas[req.scheduler_states._step_index + 1]
            collected_sigmas.append(req_sigma)
            collected_sigmas_next.append(req_sigma_next)

        sigmas_torch = torch.tensor(data=collected_sigmas).to(samples.device)
        shape = [model_outputs.shape[0]] + [1] *(model_outputs.ndim - 1)
        sigmas_torch = sigmas_torch.reshape(shape=shape)

        sigmas_next_torch = torch.tensor(data=collected_sigmas_next).to(samples.device)
        shape = [model_outputs.shape[0]] + [1] *(model_outputs.ndim - 1)
        sigmas_next_torch = sigmas_next_torch.reshape(shape=shape)

        # Update
        prev_sample = samples + (sigmas_next_torch - sigmas_torch) * model_outputs

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_outputs.dtype)

        # self._step_index += 1
        for req in runner_reqs:
            req.scheduler_states._step_index += 1

        return prev_sample