import enum

from typing import List

import torch

from sduss.scheduler import RequestStatus, Request
from sduss.model_executor.sampling_params import BaseSamplingParams

class InferenceStage(enum.Enum):
    PREPARE = enum.auto()
    DENOISING = enum.auto()
    POST = enum.auto()


class WorkerRequest:
    """Schduler's request must be converted to worker's request."""
    def __init__(
        self,
        scheduler_req: Request,
    ) -> None:
        self.request_id = scheduler_req.request_id
        # Status from new requests should be `waiting`
        self.status = scheduler_req.status
        assert self.status == RequestStatus.WAITING
        self.sampling_params = scheduler_req.sampling_params
        self.remain_steps = scheduler_req.remain_steps

        # Filled by inference procedure
        self.prepare_output = None
        self.step_input = None
        self.post_intput = None
        self.output = None

        self._initialize_sampling_params()
    
    def _initialize_sampling_params(self) -> None:
        if self.sampling_params.latents is not None:
            self.sampling_params.latents.to(torch.cuda.current_device())
        
        # TODO(MX): Other tensors are not examined.
        
        

    

class WorkerExecuteInput:
    pass

class WorkerOutput:
    def __init__(self) -> None:
        pass