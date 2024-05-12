import enum

from typing import List, Dict

import torch

from sduss.scheduler import RequestStatus, Request
from sduss.model_executor.sampling_params import BaseSamplingParams

from sduss.model_executor.diffusers.schedulers import BaseSchedulerStates
from sduss.model_executor.utils import BaseOutput

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
        # self.status = scheduler_req.status
        # assert self.status == RequestStatus.WAITING
        self.sampling_params: BaseSamplingParams = scheduler_req.sampling_params
        # self.remain_steps: int = scheduler_req.remain_steps

        # Filled by inference procedure
        self.scheduler_states: BaseSchedulerStates = None
        self.prepare_output = None
        self.step_output = None
        self.output = None

        self._initialize_sampling_params()
    
    def _initialize_sampling_params(self) -> None:
        if self.sampling_params.latents is not None:
            self.sampling_params.latents.to(torch.cuda.current_device())
        
        # TODO(MX): Other tensors are not examined.
        
# resolution -> List[request]
WorkerRequestDictType = Dict[int, List[WorkerRequest]]


class WorkerOutput:
    def __init__(
        self,
        worker_reqs: WorkerRequestDictType = None,
        status: RequestStatus = None,
    ) -> None:
        if status == RequestStatus.POSTPROCESSING:
            reqs_dict: Dict[int, BaseOutput] = {}
            for res in worker_reqs:
                for wr in worker_reqs[res]:
                    reqs_dict[wr.request_id] = wr.output
            
            # req_id -> pipeline output cls
            # pipeline output cls is assured to exist in CPU memory instead of on device
            self.req_output_dict = reqs_dict
        elif status == RequestStatus.PREPARE:
            # Return all worker requests directly
            # map: req_id -> inference steps
            reqs_dict: Dict[int, int] = {}
            for res in worker_reqs:
                for wr in worker_reqs[res]:
                    reqs_dict[wr.request_id] = len(wr.scheduler_states.timesteps)
            self.reqs_steps_dict = reqs_dict
            self.worker_reqs = worker_reqs
        else:
            raise RuntimeError