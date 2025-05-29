import time

from typing import Dict, List, TYPE_CHECKING, Optional

from .policy import Policy
from ..wrappers import SchedulerOutput
from ..utils import convert_list_to_res_dict
from ...wrappers import WorkerReqStatus

if TYPE_CHECKING:
    from sduss.dispatcher import Request

class OrcaRoundRobin(Policy):
    """ Orca scheduling implementation.

    Using a round-robin scheduling strategy.

    Features:
        Supports:
            1. Dynamic batching
        Doesn't support
            1. Mixed precision scheduling.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # All resolutions
        self.resolutions = sorted(self.request_pool.support_resolutions)

        # Set afterwards
        self._prev_res = None


    def _choose_resolution(self) -> Optional[int]:
        """If no possible choice, None will be returned."""
        if self._prev_res is None:
            for res in self.resolutions:
                if len(self.request_pool.get_unfinished_req_ids_by_res(res)) > 0:
                    self._prev_res = res
                    return res
            self._prev_res = None
            return None
        else:
            idx = self.resolutions.index(self._prev_res)
            idx = (idx + 1) % len(self.resolutions)
            last_idx = idx

            # Check next resolution
            res = self.resolutions[idx]
            if len(self.request_pool.get_unfinished_req_ids_by_res(res)) > 0:
                self._prev_res = res
                return res
            idx = (idx + 1) % len(self.resolutions)

            # Iterate until a resolution is found or we step back to last_idx
            while idx != last_idx:
                res = self.resolutions[idx]
                if len(self.request_pool.get_unfinished_req_ids_by_res(res)) > 0:
                    self._prev_res = res
                    return res
                idx = (idx + 1) % len(self.resolutions)
            # No reqs to schedule
            self._prev_res = None
            return None
    

    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Schedule requests for next iteration."""
        # 1. Pick a resolution to run
        # Also Update running resolution if no reqs in this resolution
        res = self._choose_resolution()
    
        # 2. Get reqs in this resolution to run
        # 2.1 Schedule non-denoising reqs if avaiable
        for status in [WorkerReqStatus.PREPARE, WorkerReqStatus.POSTPROCESSING]:
            scheduled_reqs = self.request_pool.get_reqs_by_complex(status=status, resolution=res)
            if len(scheduled_reqs) > 0:
                scheduled_status = status
                return SchedulerOutput(
                    scheduled_requests=convert_list_to_res_dict(scheduled_reqs),
                    status=scheduled_status,
                )

        # 2.2 Otherwise schedule denoising reqs.
        now = time.time()
        scheduled_reqs = self.request_pool.get_reqs_by_complex(status=WorkerReqStatus.DENOISING, resolution=res)
        # Always schedule the oldest reqs
        scheduled_reqs.sort(key=lambda req: now - req.arrival_time, reverse=True)
        scheduled_reqs = scheduled_reqs[:max_num]  # It's OK to be OOR(out of range)
        status = WorkerReqStatus.DENOISING
        
        return SchedulerOutput(
            scheduled_requests=convert_list_to_res_dict(scheduled_reqs),
            status=status,
        )