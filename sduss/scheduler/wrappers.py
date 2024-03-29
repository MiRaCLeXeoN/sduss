import enum

from typing import Union, Optional, List

from sduss.model_executor.sampling_params import BaseSamplingParams

class RequestStatus(enum.Enum):
    """Status of a sequence."""
    # Waiting
    WAITING = enum.auto()
    # Running
    PREPARE = enum.auto()
    DENOISING = enum.auto()
    POSTPROCESSING = enum.auto()
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

    def is_finished(self):
        return RequestStatus.is_finished(self.status)
    
    def is_compatible_with(self, req: "Request") -> bool:
        return (self.status == req.status and
                self.sampling_params.is_compatible_with(req.sampling_params))
        
    
class SchedulerOutput:    
    """Wrapper of scheduler output.
    
    Args:
        scheduled_requests: List[Request]
            Requests to run in next iteration.
        stage: RequestStatus
            The inference stage at which the selected requests are. All the
            selected requests must be in the same stage.
    """
    
    def __init__(
        self,
        scheduled_requests: List[Request],
        status: RequestStatus
    ) -> None:
        self.scheduled_requests = scheduled_requests  
        self.status = status

    
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0