import enum
import time

from typing import Union, Optional, List, TYPE_CHECKING, Dict, Set, Iterable
from collections import defaultdict

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
        self.dp_rank = None

        # Set when requst is complete
        self.output = None
        self.finish_time = None
        self.worker_arrival_time = None  # Time when the request is assigned to a worker
        self.worker_finish_time = None  # Time when the worker finishes processing the request


    def is_finished(self):
        return ReqStatus.is_finished(self.status)


    def abort(self):
        self.status = ReqStatus.ABORTED
        self.finish_time = time.time()