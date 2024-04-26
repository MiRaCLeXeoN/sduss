import enum
import time

from typing import Union, Optional, List, TYPE_CHECKING, Dict

from sduss.logger import init_logger

if TYPE_CHECKING:
    from sduss.model_executor.sampling_params import BaseSamplingParams

logger = init_logger(__name__)

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
        sampling_params: "BaseSamplingParams",
        arrival_time: Optional[float] = None,
    ):
        self.request_id = request_id
        self.sampling_params = sampling_params
        if arrival_time is None:
            self.arrival_time = time.time()

        self.status = RequestStatus.WAITING
        self.remain_steps = sampling_params.num_inference_steps

        # Set afterwards
        self.output = None
        self.finish_time = None


    def is_finished(self):
        return RequestStatus.is_finished(self.status)

    
    def is_compatible_with(self, req: "Request") -> bool:
        return (self.status == req.status and
                self.sampling_params.is_compatible_with(req.sampling_params))
        

# req_id -> req
RequestDictType = Dict[int, Request]
# resolution -> requst_dict
SchedulerOutputReqsType = Dict[int, RequestDictType]
    
class SchedulerOutput:    
    """Wrapper of scheduler output.
    
    Args:
        scheduled_requests: Dict[int, Dict[int, Request]]
            Requests to run in next iteration.
            resolution -> req_id -> req
        stage: RequestStatus
            The inference stage at which the selected requests are. All the
            selected requests must be in the same stage.
    """
    
    def __init__(
        self,
        scheduled_requests: SchedulerOutputReqsType,
        status: RequestStatus
    ) -> None:
        self.scheduled_requests: SchedulerOutputReqsType = scheduled_requests  
        self.status = status

    
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0
    
    
    def get_req_ids(self) -> List[int]:
        req_ids = []
        for reqs_dict in self.scheduled_requests.values():
            for req_id in reqs_dict.keys():
                req_ids.append(req_id)
        return req_ids
    
    
    def get_reqs_as_list(self) -> List[Request]:
        scheduler_reqs = []
        for res in self.scheduled_requests:
            for req in self.scheduled_requests[res].values():
                scheduler_reqs.append(req)
        return scheduler_reqs


class ResolutionRequestQueue:
    
    def __init__(self, resolution: int) -> None:
        
        self.resolution = resolution
        
        # queues map: req_id -> req
        self.waiting: Dict[int, Request] = {}
        self.prepare: Dict[int, Request] = {}
        self.denoising: Dict[int, Request] = {}
        self.postprocessing: Dict[int, Request] = {}
        self.finished: Dict[int, Request] = {}

        self.queues = {
            RequestStatus.WAITING : self.waiting,
            RequestStatus.PREPARE : self.prepare,
            RequestStatus.DENOISING : self.denoising,
            RequestStatus.POSTPROCESSING : self.postprocessing,
            RequestStatus.FINISHED_STOPPED : self.finished
        }

        # req_id -> req, for fast referencce
        self.reqs_mapping: Dict[int, Request] = {}
        
        self._num_unfinished_reqs = 0

    
    def add_request(self, req: Request):
        self.waiting[req.request_id] = req
        self.reqs_mapping[req.request_id] = req
        self._num_unfinished_reqs += 1

    
    def abort_requests(self, req_ids: Union[int, List[int]]):
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        
        for req_id in req_ids:
            req = self.reqs_mapping.pop(req_id)
            self.queues[req.status].pop(req_id)

            if not RequestStatus.is_finished(req.status):
                self._num_unfinished_reqs -= 1

    
    def get_queue_by_name(self, name: str):
        if name == "waiting":
            return self.waiting
        elif name == "prepare":
            return self.prepare
        elif name == "denoising":
            return self.denoising
        elif name == "postprocessing":
            return self.postprocessing
        elif name == "finished":
            return self.finished
        else:
            raise ValueError(f"Unexpected name {name}.")
    
    
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
        else:
            raise ValueError(f"Unexpected status {status}.")

    
    def get_num_unfinished_reqs(self) -> int:
        return self._num_unfinished_reqs

    
    def get_all_unfinished_reqs(self) -> List[Request]:
        """Get all unfinifhsed reqs.

        Returns:
            List[Request]: All unfinished reqs.
        """
        ret = []
        for status, q in self.queues.items():
            if status == RequestStatus.FINISHED_STOPPED:
                continue
            ret.extend(list(q.values()))
        return ret
    
    
    def get_num_finished_reqs(self) -> int:
        return len(self.finished)
    
    
    def get_finished_reqs(self) -> List[Request]:
        return list(self.finished.values())
    
    
    def get_finished_req_ids(self) -> List[Request]:
        return list(self.finished.keys())
    

    def free_all_finished_reqs(self) -> None:
        self.finished.clear()

    
    def update_reqs_status(
        self, 
        reqs_dict: RequestDictType,
        prev_status: RequestStatus,
        next_status: RequestStatus,
    ):
        """Update requests' status from prev_status to next_status.

        Note that this function assumes that reqs_dict contains only reqs
        in this resolution queue. Callers to this function are responsible for
        examination.

        Args:
            reqs_dict (RequestDictType): Req_id -> req
            prev_status (RequestStatus): Previous status
            next_status (RequestStatus): Next status
        """
        prev_que = self.get_queue_by_status(prev_status)
        next_que = self.get_queue_by_status(next_status)

        for req_id in reqs_dict:
            req = prev_que.pop(req_id)
            next_que[req_id] = req
            req.status = next_status
        
        # If requests are finished, decrease counting
        if prev_status == RequestStatus.POSTPROCESSING:
            num = len(reqs_dict)
            self._num_unfinished_reqs -= num
    
    
    def log_status(self, return_str: bool = False):
        format_str = (f"Resolution Queue: resolution={self.resolution} \n"
                      f"Remaining reqs: {self.get_num_unfinished_reqs()}\n")
        for status, queue in self.queues.items():
            format_str += f"{status=}, req_ids={queue.keys()}\n"
        if return_str:
            return format_str
        else:
            logger.debug(format_str)