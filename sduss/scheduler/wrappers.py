import enum

from typing import Union, Optional, List, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from sduss.model_executor.sampling_params import BaseSamplingParams

class RequestStatus(enum.Enum):
    """Status of a sequence."""
    # Waiting
    WAITING = enum.auto()  # newly arrived reqs
    # Running
    PREPARE = enum.auto()           # ready for prepare stage
    DENOISING = enum.auto()         # ready for denoising stage
    POSTPROCESSING = enum.auto()    # ready for postprocessing stage
    # Swapped
    SWAPPED = enum.auto()
    # Finished
    FINISHED_STOPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_ABORTED,
        ]

    @staticmethod
    def get_next_status(status: "RequestStatus") -> Optional["RequestStatus"]:
        if status == RequestStatus.WAITING:
            return RequestStatus.PREPARE
        elif status == RequestStatus.PREPARE:
            return RequestStatus.DENOISING
        elif status == RequestStatus.DENOISING:
            return RequestStatus.POSTPROCESSING
        elif status == RequestStatus.POSTPROCESSING:
            return RequestStatus.FINISHED_STOPPED
        elif status == RequestStatus.SWAPPED:
            # We cannot decide here, leave for further processing
            return None
        else:
            raise RuntimeError("We cannot decide next status.")

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> Union[str, None]:
        if status == RequestStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == RequestStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        else:
            finish_reason = None
        return finish_reason


class Request:
    """Request wrapper. Used in engine and schduler for scheduling."""
    def __init__(
        self,
        request_id: int,
        arrival_time: float,
        sampling_params: BaseSamplingParams,
    ):
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.sampling_params = sampling_params

        self.status = RequestStatus.WAITING
        self.remain_steps = sampling_params.num_inference_steps

    def is_finished(self):
        return RequestStatus.is_finished(self.status)
    
    def is_compatible_with(self, req: "Request") -> bool:
        return (self.status == req.status and
                self.sampling_params.is_compatible_with(req.sampling_params))
        
    
class SchedulerOutput:    
    """Wrapper of scheduler output.
    
    Args:
        scheduled_requests: Dict[int, Dict[int, Request]]
            Requests to run in next iteration.
        stage: RequestStatus
            The inference stage at which the selected requests are. All the
            selected requests must be in the same stage.
    """
    
    def __init__(
        self,
        scheduled_requests: Dict[int, Dict[int, Request]],
        status: RequestStatus
    ) -> None:
        self.scheduled_requests = scheduled_requests  
        self.status = status

    
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0


ResolutionQueueType = Dict[int, Request]

class ResolutionRequestQueue:
    
    def __init__(self, resolution: int) -> None:
        
        self.resolution = resolution
        
        # queues map: req_id -> req
        self.waiting = {}
        self.prepare = {}
        self.denoising = {}
        self.postprocessing = {}
        self.finished = {}

        self.queues = {
            "waiting" : self.waiting,
            "prepare" : self.prepare,
            "denoising" : self.denoising,
            "postprocessing" : self.postprocessing,
            "finished" : self.finished
        }
        
        self._num_unfinished_reqs = 0

    
    def add_request(self, req: Request):
        self.waiting[req.request_id] = req
        self._num_unfinished_reqs += 1

    
    def abort_requests(self, req_ids: Union[int, List[int]]):
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        
        for req_id in req_ids:
            for queue in self.queues.values():
                if queue.pop(req_id, None) is not None:
                    self._num_unfinished_reqs -= 1
                    break

    
    def get_num_unfinished_reqs(self) -> int:
        return self._num_unfinished_reqs
    
    
    def get_queue_by_name(self, name: str):
        return self.queues[name]
    
    
    def get_queue_by_status(self, status: RequestStatus) -> Dict[int, Request]:
        if status == RequestStatus.WAITING:
            return self.waiting
        elif status == RequestStatus.PREPARE:
            return self.prepare
        elif status == RequestStatus.DENOISING:
            return self.denoising
        elif status == RequestStatus.POSTPROCESSING:
            return self.postprocessing
        elif status == RequestStatus.FINISHED_STOPPED:
            return self.finished
    
    
    def get_all_unfinished_reqs(self) -> List[Request]:
        """Get all unfinifhsed reqs.

        Returns:
            List[Request]: All unfinished reqs.
        """
        ret = []
        for name, q in self.queues.items():
            if name == "finished":
                continue
            ret.extend(q.values())
        return ret
    
    
    def get_fnished_req_ids(self) -> List[int]:
        return list(self.finished.keys())
    
    
    def free_all_finished_reqs(self) -> None:
        self.finished.clear()

    
            