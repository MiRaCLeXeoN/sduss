import enum
import time

from typing import Union, Optional, List, TYPE_CHECKING, Dict
from datetime import datetime

from sduss.logger import init_logger

from .esymred_utils import (DISCARD_SLACK, DENOISING_DDL, POSTPROCESSING_DDL, STANDALONE,
                            Hyper_Parameter)

if TYPE_CHECKING:
    from sduss.model_executor.sampling_params import BaseSamplingParams

logger = init_logger(__name__)

class RequestStatus(enum.IntEnum):
    """Status of a sequence."""
    # Empty
    EMPTY = enum.auto()             # Use by the scheduler to indicate no requests to run
    # Waiting
    WAITING = enum.auto()           # newly arrived reqs
    # Running
    PREPARE = enum.auto()           # ready for prepare stage
    DENOISING = enum.auto()         # ready for denoising stage
    POSTPROCESSING = enum.auto()    # ready for postprocessing stage
    # Finished
    FINISHED_STOPPED = enum.auto()
    # Exception
    EXCEPTION_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.FINISHED_STOPPED,
        ]
    
    
    @staticmethod
    def is_exception(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.EXCEPTION_ABORTED,
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
        else:
            raise RuntimeError("We cannot decide next status.")

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> Union[str, None]:
        if status == RequestStatus.FINISHED_STOPPED:
            finish_reason = "stop"
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

        self.status = RequestStatus.WAITING
        self.remain_steps = sampling_params.num_inference_steps

        # Set afterwards
        self.output = None
        self.finish_time = None

        # used by esymred
        self.start_denoising = False
        self.is_discard = False
        # Predict time indicates the time estimated to run until complete all
        # Unet iterations, with respect to the current workload.
        self.predict_time = None


    def is_finished(self):
        return RequestStatus.is_finished(self.status)


    def update_predict_time(self, predict_time:float):
        self.predict_time = predict_time


    def set_slack(
        self, 
        model_name : str, 
        is_running : bool, 
        current_running_time_cost : float,
    ):
        # If discarded, return directly
        if self.is_discard:
            self.slack = DISCARD_SLACK
            return

        resolution = self.sampling_params.resolution
        status = self.status
        # Get ddl
        if status == RequestStatus.WAITING or status == RequestStatus.PREPARE:
            self.slack = 0
            return 
        elif status == RequestStatus.DENOISING:
            stage = "denoising"
            ddl = DENOISING_DDL[model_name][str(resolution)]
        elif status == RequestStatus.POSTPROCESSING:
            stage = "postprocessing"
            ddl = POSTPROCESSING_DDL[model_name][str(resolution)]
        
        unit_unet_time = STANDALONE[model_name][stage][str(resolution)]
        if stage == "postprocessing":
            self.slack = (ddl - unit_unet_time - current_running_time_cost - (time.time() - self.arrival_time)
                            ) / (unit_unet_time * Hyper_Parameter[model_name][stage][str(resolution)])
        elif stage == "denoising":
            if is_running:
                # Suppose we have started at least one round
                self.slack = (ddl - self.predict_time - current_running_time_cost - (time.time() - self.arrival_time)
                                ) / unit_unet_time
            else:
                # Denoising not started yet
                self.slack = (ddl - unit_unet_time - current_running_time_cost - (time.time() - self.arrival_time)
                                ) / unit_unet_time
        self.remain_time = ddl - current_running_time_cost - (time.time() - self.arrival_time)

    
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
        status: RequestStatus
            The inference stage at which the selected requests are. All the
            selected requests must be in the same stage.
        prepare_requests: Dict[int, Dict[int, Request]]
            Overlapped prepare-stage requets. If status is prepare, this must be none.
    """
    
    def __init__(
        self,
        scheduled_requests: SchedulerOutputReqsType = None,
        status: RequestStatus = None,
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
        if mixed_precision and self.status == RequestStatus.DENOISING:
            assert self.is_sliced is not None and self.patch_size is not None
        # Check prepare_requests. It should not co-exist with RequestStatus.PREPARE
        if self.status == RequestStatus.PREPARE:
            assert self.prepare_requests is None

    
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0
    
    
    def has_prepare_requests(self) -> bool:
        return self.prepare_requests
    
    
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
        scheduler_reqs = []
        for res in self.prepare_requests:
            for req in self.prepare_requests[res].values():
                scheduler_reqs.append(req)
        return scheduler_reqs
    
    
    def get_log_string(self) -> str:
        ret = f"status: {self.status}\n"
        if self.scheduled_requests:
            ret += f"scheduled reqs: \n"
            for res in self.scheduled_requests:
                ret += f"{res=}  reqs: "
                for req_id in self.scheduled_requests[res]:
                    ret += "%d," % req_id
                ret += "\n"
        if self.prepare_requests:
            ret += "overlapped prepare reqs: \n"
            for res in self.scheduled_requests:
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
        self.aborted: Dict[int, Request] = {}
        self.queues = {
            RequestStatus.WAITING : self.waiting,
            RequestStatus.PREPARE : self.prepare,
            RequestStatus.DENOISING : self.denoising,
            RequestStatus.POSTPROCESSING : self.postprocessing,
            RequestStatus.FINISHED_STOPPED : self.finished,
            RequestStatus.EXCEPTION_ABORTED: self.aborted
        }

        # req_id -> req, for fast referencce
        self.reqs_mapping: Dict[int, Request] = {}
        
        self._num_unfinished_normal_reqs = 0

    
    def add_request(self, req: Request):
        self.waiting[req.request_id] = req
        self.reqs_mapping[req.request_id] = req
        self._num_unfinished_normal_reqs += 1

    
    def abort_requests(self, req_ids: Union[int, List[int]]):
        """This method abort requests.
        
        Aborted requests will not be released immediately. They will be
        marked as aborted and update to FINISHED_ABORTED status, with all
        belonging data kept. So the aborted requests can be recovered if
        necessary.

        To really release a request, call `remove_requests` method after 
        calling this method.
        """
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        
        for req_id in req_ids:
            # We don't remove reference until "remove_reqs" is called
            req = self.reqs_mapping[req_id]
            self.queues[req.status].pop(req_id)
            self.aborted[req_id] = req

            if not RequestStatus.is_finished(req.status):
                self._num_unfinished_normal_reqs -= 1


    def recover_aborted_requests(self):
        for req_id, req in self.aborted.items():
            self.queues[req.status][req_id] = req
            if (not RequestStatus.is_finished(req.status) and
                not RequestStatus.is_exception(req.status)):
                self._num_unfinished_normal_reqs += 1


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
        elif name == "aborted":
            return self.aborted
        else:
            raise ValueError(f"Unexpected name {name}.")
    
    
    def _get_queue_by_status(self, status: RequestStatus) -> Dict[int, Request]:
        """This method should only be invoked by methods of this class."""
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
        elif status == RequestStatus.EXCEPTION_ABORTED:
            return self.aborted
        else:
            raise ValueError(f"Unexpected status {status}.")


    def get_queue_by_status(self, status: RequestStatus) -> Dict[int, Request]:
        """This method will return a shallow copy of the queue dict to
        prevent any possible outside interruption."""
        if status == RequestStatus.WAITING:
            return self.waiting.copy()
        elif status == RequestStatus.PREPARE:
            return self.prepare.copy()
        elif status == RequestStatus.DENOISING:
            return self.denoising.copy()
        elif status == RequestStatus.POSTPROCESSING:
            return self.postprocessing.copy()
        elif status == RequestStatus.FINISHED_STOPPED:
            return self.finished.copy()
        elif status == RequestStatus.EXCEPTION_ABORTED:
            return self.aborted.copy()
        else:
            raise ValueError(f"Unexpected status {status}.")
    
    
    def get_all_reqs_by_status(self, status: RequestStatus) -> List[Request]:
        return list(self.queues[status].values())
    
    
    def get_num_reqs_by_staus(self, status: RequestStatus) -> int:
        return len(self.queues[status])

    
    def get_num_unfinished_normal_reqs(self) -> int:
        return self._num_unfinished_normal_reqs

    
    def get_num_unfreed_normal_reqs(self) -> int:
        num = 0
        for status, q in self.queues.items():
            if not RequestStatus.is_exception(status):
                num += len(q)
        return num

    
    def get_all_unfinished_normal_reqs(self) -> List[Request]:
        """Get all unfinifhsed reqs.

        Returns:
            List[Request]: All unfinished reqs.
        """
        ret = []
        for status, q in self.queues.items():
            if (RequestStatus.is_finished(status) or 
                RequestStatus.is_exception(status)):
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

            if not RequestStatus.is_finished(req.status):
                self._num_unfinished_normal_reqs -= 1

    
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
        prev_que = self._get_queue_by_status(prev_status)
        next_que = self._get_queue_by_status(next_status)

        for req_id in reqs_dict:
            req = prev_que.pop(req_id)
            next_que[req_id] = req
            req.status = next_status
        
        # If requests are finished, decrease counting
        if RequestStatus.is_finished(next_status):
            num = len(reqs_dict)
            self._num_unfinished_normal_reqs -= num
    
    
    def update_all_waiting_reqs_to_prepare(self) -> None:
        """Update all waiting reqs to prepare status."""
        for req_id in self.waiting:
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