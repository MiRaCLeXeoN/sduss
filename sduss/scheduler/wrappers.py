import enum

from typing import Union

from sduss.model_executor.sampling_params import BaseSamplingParams

class InferenceStage(enum.Enum):
    WAITING = enum.auto()
    PREPARE = enum.auto()
    DENOISING = enum.auto()
    POST = enum.auto()
    FINISHED = enum.auto()

class RequestStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_ABORTED,
        ]

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
        self.inference_stage = InferenceStage.WAITING

    def is_finished(self):
        return RequestStatus.is_finished(self.status)
        
    