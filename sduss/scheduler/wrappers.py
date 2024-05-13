import enum
import time

from typing import Union, Optional, List, TYPE_CHECKING, Dict

from sduss.logger import init_logger

if TYPE_CHECKING:
    from sduss.model_executor.sampling_params import BaseSamplingParams

logger = init_logger(__name__)
DISCARD_SLACK = 100
DENOISING_DDL = {
    "sd1.5": {
        "256": (3.1356 + 0.0394) * 5 * 0.95,
        "512": (3.1788 + 0.06974) * 5 * 0.95,
        "768": (3.87 + 0.1363) * 5 * 0.95,
    },
    "sdxl": {
        "512": (8.587 + 0.13413) * 5 * 0.95,
        "768": (8.6663 + 0.24094) * 5 * 0.95,
        "1024": (8.7663 + 0.434) * 5 * 0.95,
    }
}

POSTPROCESSING_DDL = {
    "sd1.5": {
        "256": (3.1356 + 0.0394) * 5,
        "512": (3.1788 + 0.06974) * 5,
        "768": (3.87 + 0.1363) * 5,
    },
    "sdxl": {
        "512": (8.587 + 0.13413) * 5,
        "768": (8.6663 + 0.24094) * 5,
        "1024": (8.7663 + 0.434) * 5,
    }
}

STANDALONE = {
    "sd1.5": {
        "denoising": {
            "256": 3.1356,
            "512": 3.1788,
            "768": 3.87,
        },
        "postprocessing": {
            "256": 0.0394,
            "512": 0.06974,
            "768": 0.1363,
        }
    },
    "sdxl": {
        "denoising": {
            "512": 8.587,
            "768": 8.6663,
            "1024": 8.7663,
        },
        "postprocessing": {
            "512": 0.13413,
            "768": 0.24094,
            "1024": 0.434,
        }
    }
}

Hyper_Parameter = {
    "sd1.5": {
        "postprocessing": {
            "256": 4,
            "512": 2,
            "768": 1,
        }
    },
    "sdxl": {
        "postprocessing": {
            "512": 4,
            "768": 2,
            "1024": 1,
        }
    },
    "get_best_tp_th": 1,
    "active_queue_timeout_th": 0.1,
}

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
        self.start_denoising = False
        self.is_discard = False


    def is_finished(self):
        return RequestStatus.is_finished(self.status)

    def update_predict_time(self, predict_time:float):
        self.predict_time = predict_time

    def get_slack(self, model_name:str, is_running:bool, current_running_time_cost:float):
        if self.is_discard:
            self.slack = DISCARD_SLACK
            return
        resolution = self.sampling_params.resolution
        runtime = STANDALONE[model_name][str(resolution)]
        status = self.status
        if status == RequestStatus.DENOISING:
            stage = "denoising"
            ddl = DENOISING_DDL[model_name][str(resolution)]
        elif status == RequestStatus.POSTPROCESSING:
            stage = "postprocessing"
            ddl = POSTPROCESSING_DDL[model_name][str(resolution)]
        runtime = STANDALONE[model_name][stage][str(resolution)]
        if status == RequestStatus.PREPARE:
            # prepare阶段直接开始在CPU执行
            self.slack = 0
        else:
            if stage == "postprocessing":
                self.slack = (ddl - runtime - current_running_time_cost - (time.time() - self.arrival_time)) / (runtime * Hyper_Parameter[model_name][str(resolution)])
            elif stage == "denoising":
                if is_running:
                    # 计算中的denoising阶段
                    self.slack = (ddl - self.predict_time - current_running_time_cost - (time.time() - self.arrival_time)) / runtime
                else:
                    # 等待中的denoising阶段
                    self.slack = (ddl - runtime - current_running_time_cost - (time.time() - self.arrival_time)) / runtime
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
        **kwargs,
    ) -> None:
        self.scheduled_requests: SchedulerOutputReqsType = scheduled_requests  
        self.prepare_requests: SchedulerOutputReqsType = prepare_requests
        self.status = status

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
        return self.prepare_requests is not None
    
    
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
    
    
    def get_all_reqs_by_status(self, status: RequestStatus) -> List[Request]:
        return list(self.queues[status].values())

    
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
    
    
    def free_finished_reqs(self, reqs: Union[List[Request], Request]) -> None:
        if isinstance(reqs, Request):
            reqs = [reqs]
        for req in reqs:
            self.finished.pop(req.request_id)
    
    
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