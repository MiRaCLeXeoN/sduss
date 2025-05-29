import enum

from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from sduss.model_executor.sampling_params import BaseSamplingParams
    from sduss.model_executor.diffusers import BasePipelinePrepareOutput
    from sduss.model_executor.diffusers import BaseSchedulerStates
    from sduss.model_executor.utils import BaseOutput
    from ..wrappers import WorkerRequest


class InferenceStage(enum.Enum):
    PREPARE = enum.auto()
    DENOISING = enum.auto()
    POST = enum.auto()


class RunnerRequest:
    """Schduler's request must be converted to worker's request."""
    def __init__(
        self,
        req_id,
        req_sp
    ) -> None:
        self.request_id = req_id
        # Status from new requests should be `waiting`
        # self.status = scheduler_req.status
        self.sampling_params: "BaseSamplingParams" = req_sp
        # self.remain_steps: int = scheduler_req.remain_steps

        # Filled by inference procedure
        self.scheduler_states: "BaseSchedulerStates" = None
        self.prepare_output: "BasePipelinePrepareOutput" = None
        self.step_output = None
        self.output = None


# resolution -> List[request]
RunnerRequestDictType = Dict[int, List[RunnerRequest]]


class RunnerOutput:
    def __init__(
        self,
        runner_reqs: RunnerRequestDictType,
        stage: InferenceStage = None,
        start_time : float = None,
        end_time : float = None,
    ) -> None:
        # Performance recording
        self.start_time = start_time
        self.end_time = end_time

        # Set by specific stages
        self.req_output_dict = None

        if stage == InferenceStage.POST:
            reqs_dict: 'Dict[int, BaseOutput]' = {}
            for res in runner_reqs:
                for wr in runner_reqs[res]:
                    reqs_dict[wr.request_id] = wr.output
            
            # req_id -> pipeline output cls
            # pipeline output cls is assured to exist in CPU memory instead of on device
            self.req_output_dict = reqs_dict
        elif stage == InferenceStage.PREPARE:
            pass
        elif stage == InferenceStage.DENOISING:
            pass