import enum
import time

from typing import Union, Optional, List, TYPE_CHECKING, Dict
from datetime import datetime

from sduss.logger import init_logger

if TYPE_CHECKING:
    from sduss.model_executor.sampling_params import BaseSamplingParams

logger = init_logger(__name__)

class ReqStatus(enum.IntEnum):
    """Status of a sequence."""
    # Waiting
    WAITING = enum.auto()           # newly arrived reqs
    # Running
    RUNNING = enum.auto()
    # Finished
    FINISHED = enum.auto()
    # Exception
    ABORTED = enum.auto()
    EXCEPTION_SWAPPED = enum.auto()

    @staticmethod
    def is_finished(status: "ReqStatus") -> bool:
        return status in [
            ReqStatus.FINISHED,
            ReqStatus.ABORTED
        ]
    
    
    @staticmethod
    def is_normal_finished(status: "ReqStatus") -> bool:
        return status in [
            ReqStatus.FINISHED,
        ]
    
    
    @staticmethod
    def is_exception(status: "ReqStatus") -> bool:
        return status in [
            ReqStatus.EXCEPTION_SWAPPED,
        ]
    
    
    @staticmethod
    def is_running(status: "ReqStatus") -> bool:
        return status in [
            ReqStatus.RUNNING,
        ]
    

    @staticmethod
    def get_finished_reason(status: "ReqStatus") -> Union[str, None]:
        if status == ReqStatus.FINISHED:
            finish_reason = "finished normally"
        elif status == ReqStatus.ABORTED:
            finish_reason = "aborted"
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
        else:
            self.arrival_time = arrival_time
        
        # Used by dispatcher
        self.status = ReqStatus.WAITING
        self.device_num = None
        self.dp_rank = None

        # Set when requst is complete
        self.output = None
        self.finish_time = None


    def is_finished(self):
        return ReqStatus.is_finished(self.status)


    def abort(self):
        self.status = ReqStatus.ABORTED
        self.finish_time = time.time()


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
        status: DispatcherRequestStatus
            The inference stage at which the selected requests are. All the
            selected requests must be in the same stage.
        prepare_requests: Dict[int, Dict[int, Request]]
            Overlapped prepare-stage requets. If status is prepare, this must be none.
    """
    
    def __init__(
        self,
        scheduled_requests: SchedulerOutputReqsType = None,
        status: ReqStatus = None,
        prepare_requests: SchedulerOutputReqsType = None,
        abort_req_ids: List[int] = None,
        **kwargs,
    ) -> None:
        self.scheduled_requests: SchedulerOutputReqsType = scheduled_requests  
        self.prepare_requests: SchedulerOutputReqsType = prepare_requests
        self.abort_req_ids: List[int] = abort_req_ids
        self.status = status

        # This is a bit hacky, since we jump over the normal procedure
        # of updating from WAITING to PREPARE
        # This is an option to fast forward WAITING reqs
        self.update_all_waiting_reqs: bool = kwargs.pop("update_all_waiting_reqs", False)

        self.is_sliced = kwargs.pop("is_sliced", None)
        self.patch_size = kwargs.pop("patch_size", None)

        # Check up
        self._verify_params()

    
    def _verify_params(self):
        # mixed precision
        mixed_precision = len(self.scheduled_requests) > 1
        if mixed_precision and self.status == ReqStatus.DENOISING:
            assert self.is_sliced is not None and self.patch_size is not None
        # Check prepare_requests. It should not co-exist with DispatcherRequestStatus.PREPARE
        if self.status == ReqStatus.PREPARE:
            assert self.prepare_requests is None

    
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0
    
    
    def has_prepare_requests(self) -> bool:
        return self.prepare_requests and len(self.prepare_requests) > 0
    
    
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
    
    
    def get_prepare_reqs_as_list(self) -> List[Request]:
        if self.prepare_requests is None:
            return []
        prepare_reqs = []
        for res in self.prepare_requests:
            for req in self.prepare_requests[res].values():
                prepare_reqs.append(req)
        return prepare_reqs
    
    
    def get_log_string(self) -> str:
        ret = f"status: {str(self.status)}\n"
        if self.scheduled_requests:
            ret += f"scheduled reqs: \n"
            for res in self.scheduled_requests:
                ret += f"{res=}  reqs: "
                for req_id in self.scheduled_requests[res]:
                    ret += "%d," % req_id
                ret += "\n"
        if self.prepare_requests:
            ret += "overlapped prepare reqs: \n"
            for res in self.prepare_requests:
                ret += f"{res=}  reqs: "
                for req_id in self.prepare_requests[res]:
                    ret += "%d," % req_id
                ret += "\n"
        return ret


class ResolutionRequestQueue:
    
    def __init__(self, resolution: int) -> None:
        
        self.resolution = resolution
        
        # queues map: req_id -> req
        self.waiting: Dict[int, Request] = {}
        self.prepare: Dict[int, Request] = {}
        self.denoising: Dict[int, Request] = {}
        self.postprocessing: Dict[int, Request] = {}
        self.finished: Dict[int, Request] = {}
        self.swapped: Dict[int, Request] = {}
        self.executing: Dict[int, Request] = {}
        self.queues = {
            ReqStatus.WAITING : self.waiting,
            ReqStatus.PREPARE : self.prepare,
            ReqStatus.DENOISING : self.denoising,
            ReqStatus.POSTPROCESSING : self.postprocessing,
            ReqStatus.FINISHED_STOPPED : self.finished,
            ReqStatus.EXCEPTION_SWAPPED: self.swapped,
            ReqStatus.EXECUTING : self.executing,
        }

        # req_id -> req, for fast referencce
        self.reqs_mapping: Dict[int, Request] = {}
        
        self._num_unfinished_normal_reqs = 0

    
    def add_request(self, req: Request):
        self.waiting[req.request_id] = req
        self.reqs_mapping[req.request_id] = req
        self._num_unfinished_normal_reqs += 1

    
    def abort_requests(self, req_ids: Union[int, List[int]]):
        """This method abort requests and untrack them."""
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        
        for req_id in req_ids:
            req = self.reqs_mapping.pop(req_id)
            self.queues[req.status].pop(req_id)

            if not ReqStatus.is_finished(req.status):
                self._num_unfinished_normal_reqs -= 1
            req.abort()


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
        elif name == "swapped":
            return self.swapped
        elif name == "executing":
            return self.executing
        else:
            raise ValueError(f"Unexpected name {name}.")
    
    
    def _get_queue_by_status(self, status: ReqStatus) -> Dict[int, Request]:
        """This method should only be invoked by methods of this class."""
        if status == ReqStatus.WAITING:
            return self.waiting
        elif status == ReqStatus.PREPARE:
            return self.prepare
        elif status == ReqStatus.DENOISING:
            return self.denoising
        elif status == ReqStatus.POSTPROCESSING:
            return self.postprocessing
        elif status == ReqStatus.FINISHED_STOPPED:
            return self.finished
        elif status == ReqStatus.EXCEPTION_SWAPPED:
            return self.swapped
        elif status == ReqStatus.EXECUTING:
            return self.executing
        else:
            raise ValueError(f"Unexpected status {status}.")


    def get_queue_by_status(self, status: ReqStatus) -> Dict[int, Request]:
        """This method will return a shallow copy of the queue dict to
        prevent any possible outside interruption."""
        if status == ReqStatus.WAITING:
            return self.waiting.copy()
        elif status == ReqStatus.PREPARE:
            return self.prepare.copy()
        elif status == ReqStatus.DENOISING:
            return self.denoising.copy()
        elif status == ReqStatus.POSTPROCESSING:
            return self.postprocessing.copy()
        elif status == ReqStatus.FINISHED_STOPPED:
            return self.finished.copy()
        elif status == ReqStatus.EXCEPTION_SWAPPED:
            return self.swapped.copy()
        elif status == ReqStatus.EXECUTING:
            return self.executing.copy()
        else:
            raise ValueError(f"Unexpected status {str(status)}.")
    
    
    def get_all_reqs_by_status(self, status: ReqStatus) -> List[Request]:
        return list(self.queues[status].values())
    
    
    def get_num_reqs_by_staus(self, status: ReqStatus) -> int:
        return len(self.queues[status])

    
    def get_num_unfinished_normal_reqs(self) -> int:
        return self._num_unfinished_normal_reqs

    
    def get_num_unfreed_normal_reqs(self) -> int:
        num = 0
        for status, q in self.queues.items():
            if not ReqStatus.is_exception(status):
                num += len(q)
        return num

    
    def get_all_unfinished_normal_reqs(self) -> List[Request]:
        """Get all unfinifhsed reqs.

        Returns:
            List[Request]: All unfinished reqs.
        """
        ret = []
        for status, q in self.queues.items():
            if (ReqStatus.is_executing(status) or
                ReqStatus.is_finished(status) or 
                ReqStatus.is_exception(status)):
                continue
            ret.extend(list(q.values()))
        return ret
    
    
    def get_num_finished_reqs(self) -> int:
        return len(self.finished)

    
    def get_num_unfreed_reqs(self) -> int:
        return len(self.reqs_mapping.keys())
    
    
    def get_finished_reqs(self) -> List[Request]:
        return list(self.finished.values())
    
    
    def get_finished_req_ids(self) -> List[Request]:
        return list(self.finished.keys())
    

    def free_all_finished_reqs(self) -> None:
        for req_id in self.finished:
            self.reqs_mapping.pop(req_id)
        self.finished.clear()
    
    
    def free_finished_reqs(self, req_ids: Union[List[int], int]) -> None:
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        for req_id in req_ids:
            self.reqs_mapping.pop(req_id)
            self.finished.pop(req_id)


    def free_reqs(self, req_ids: Union[int, List[int]]):
        """This method free requests from the scheduler."""
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        
        for req_id in req_ids:
            req = self.reqs_mapping[req_id]
            self.queues[req.status].pop(req_id)

            if not ReqStatus.is_finished(req.status):
                self._num_unfinished_normal_reqs -= 1

    
    def update_reqs_status(
        self, 
        reqs_dict: RequestDictType,
        prev_status: ReqStatus,
        next_status: ReqStatus,
    ):
        """Update requests' status from prev_status to next_status.

        Note that this function assumes that reqs_dict contains only reqs
        in this resolution queue. Callers to this function are responsible for
        examination.

        Args:
            reqs_dict (RequestDictType): Req_id -> req
            prev_status (DispatcherRequestStatus): Previous status
            next_status (DispatcherRequestStatus): Next status
        """
        prev_que = self._get_queue_by_status(prev_status)
        next_que = self._get_queue_by_status(next_status)

        for req_id in reqs_dict:
            req = prev_que.pop(req_id)
            next_que[req_id] = req
            req.status = next_status
        
        # If requests are finished, decrease counting
        if ReqStatus.is_finished(next_status):
            num = len(reqs_dict)
            self._num_unfinished_normal_reqs -= num
    
    
    def update_all_waiting_reqs_to_prepare(self) -> None:
        """Update all waiting reqs to prepare status."""
        for req_id in self.waiting:
            self.reqs_mapping[req_id].status = ReqStatus.PREPARE
            self.prepare[req_id] = self.waiting[req_id]
        self.waiting.clear()
    
    
    def log_status(self, return_str: bool = False):
        format_str = (f"Resolution Queue: resolution={self.resolution} \n"
                      f"Remaining reqs: {self.get_num_unfinished_normal_reqs()}\n")
        for status, queue in self.queues.items():
            format_str += f"{status=}, req_ids={queue.keys()}\n"
        if return_str:
            return format_str
        else:
            logger.debug(format_str)