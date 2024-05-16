import time

from typing import Dict, List, TYPE_CHECKING

from sduss.scheduler.wrappers import ResolutionRequestQueue, RequestStatus

from .policy import Policy
from ..wrappers import SchedulerOutput
from ..utils import convert_list_to_res_dict

if TYPE_CHECKING:
    from sduss.scheduler import Request

class OrcaResByRes(Policy):
    """ Orca scheduling implementation.

    Using a resolution-by-resolution scheduling strategy.

    Features:
        Supports:
            1. Dynamic batching
        Doesn't support
            1. Mixed precision scheduling.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # All resolutions
        self.resolutions = list(self.request_pool.keys()).sort()

        # Set afterwards
        self._running_res = None


    def _choose_resolution(self) -> int:
        if (self._running_res is not None 
            and self.request_pool[self._running_res].get_num_unfinished_normal_reqs() > 0):
            # We have more reqs of the current resolution
            return self._running_res

        sche_res = None
        for res in self.resolutions:
            if self.request_pool[res].get_num_unfinished_normal_reqs() > 0:
                sche_res = res
        # If not schedules at all, we will return None
        self._running_res = sche_res
        return sche_res
    

    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Schedule requests for next iteration."""
        # 1. Pick a resolution to run
        # Also Update running resolution if no reqs in this resolution
        res = self._choose_resolution()
        if res is None:
            # No reqs to schedule
            return SchedulerOutput(
                scheduled_requests={},
                status=RequestStatus.EMPTY,
            )
    
        # 2. Get reqs in this resolution to run
        resolution_queue = self.request_pool[res]
        # 2.1 Schedule non-denoising reqs if avaiable
        for status in [RequestStatus.WAITING, RequestStatus.PREPARE, RequestStatus.POSTPROCESSING]:
            scheduled_reqs = resolution_queue.get_all_reqs_by_status(status)
            if len(scheduled_reqs) > 0:
                scheduled_status = status
                return SchedulerOutput(
                    scheduled_requests=convert_list_to_res_dict(scheduled_reqs),
                    status=scheduled_status,
                )
        # 2.2 Otherwise schedule denoising reqs.
        now = time.time()
        scheduled_reqs = resolution_queue.get_all_reqs_by_status(RequestStatus.DENOISING)
        # Always schedule the oldest reqs
        scheduled_reqs.sort(key=lambda req: now - req.arrival_time, reverse=True)
        scheduled_reqs = scheduled_reqs[:max_num]  # It's OK to be OOR(out of range)
        status = RequestStatus.DENOISING
        
        return SchedulerOutput(
            scheduled_requests=convert_list_to_res_dict(scheduled_reqs),
            status=status,
        )