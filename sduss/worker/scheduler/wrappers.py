from typing import Union, Optional, List, TYPE_CHECKING, Dict

from sduss.logger import init_logger
from ..wrappers import WorkerReqStatus, WorkerRequest

if TYPE_CHECKING:
    from sduss.model_executor.sampling_params import BaseSamplingParams

logger = init_logger(__name__)

# req_id -> req
RequestDictType = Dict[int, WorkerRequest]
# resolution -> requst_dict
SchedulerOutputReqsType = Dict[int, RequestDictType]
    
class SchedulerOutput:    
    """Wrapper of scheduler output.
    
    Args:
        scheduled_requests: Dict[int, Dict[int, WorkerRequest]]
            Requests to run in next iteration.
            resolution -> req_id -> req
        status: ReqStatus
            The inference stage at which the selected requests are. All the
            selected requests must be in the same stage.
        prepare_requests: Dict[int, Dict[int, WorkerRequest]]
            Overlapped prepare-stage requets. If status is prepare, this must be none.
    """
    
    def __init__(
        self,
        scheduled_requests: SchedulerOutputReqsType = None,
        status: WorkerReqStatus = None,
        abort_req_ids: List[int] = None,
        **kwargs,
    ) -> None:
        self.scheduled_requests: SchedulerOutputReqsType = scheduled_requests  
        self.abort_req_ids: List[int] = abort_req_ids
        self.status = status

        self.is_sliced = kwargs.pop("is_sliced", None)
        self.patch_size = kwargs.pop("patch_size", None)

        # Check up
        self._verify_params()

    
    def _verify_params(self):
        # mixed precision
        mixed_precision = len(self.scheduled_requests) > 1
        if mixed_precision and self.status == WorkerReqStatus.DENOISING:
            assert self.is_sliced is not None and self.patch_size is not None

    
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0
    
    
    def get_req_ids(self) -> List[int]:
        req_ids = []
        for reqs_dict in self.scheduled_requests.values():
            req_ids.extend(list(reqs_dict.keys()))
        return req_ids
    
    
    def get_reqs_as_list(self) -> List[WorkerRequest]:
        reqs = []
        for reqs_dict in self.scheduled_requests.values():
            reqs.extend(list(reqs_dict.values()))
        return reqs
    
    
    def get_log_string(self) -> str:
        ret = f"status: {str(self.status)}\n"
        if self.scheduled_requests:
            ret += f"scheduled reqs: \n"
            for res in self.scheduled_requests:
                ret += f"{res=}  reqs: "
                for req_id in self.scheduled_requests[res]:
                    ret += "%d," % req_id
                ret += "\n"
        return ret