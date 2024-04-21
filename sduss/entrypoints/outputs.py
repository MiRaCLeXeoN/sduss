import time
import datetime

from typing import List, Optional, Type

from sduss.scheduler import Request, RequestStatus

class RequestOutput:
    """The output wrapper of a request.
    
    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """
    def __init__(
        self,
        request: Request
    ) -> None:
        self.request_id = request.request_id
        self.output = request.output
        self.finished = request.status == RequestStatus.FINISHED_STOPPED

        self.start_datetime = datetime.datetime.fromtimestamp(request.arrival_time)
        self.finish_datetime = datetime.datetime.fromtimestamp(request.finish_time)
        self.time_consumption = request.finish_time - request.arrival_time

        
    def __repr__(self) -> str:
        return (f"start time={self.start_datetime}\n"
                f"finish time={self.finish_datetime}\n"
                f"time consumption={self.time_consumption:.3f}s")