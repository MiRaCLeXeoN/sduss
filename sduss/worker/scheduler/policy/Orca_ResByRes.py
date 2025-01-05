import time

from typing import Dict, List, TYPE_CHECKING

from .policy import Policy
from ..wrappers import SchedulerOutput
from ..utils import convert_list_to_res_dict
from ...wrappers import WorkerReqStatus

from sduss.logger import init_logger

logger = init_logger(__name__)

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
        self.resolutions = sorted(self.request_pool.support_resolutions)

        # Set afterwards
        self._running_res = None


    def _choose_resolution(self) -> int:
        if self._running_res is not None:
            # Check if we have unfinished reqs of this resolution
            req_ids = self.request_pool.get_unfinished_req_ids_by_res(self._running_res)
            if len(req_ids) > 0:
                # We have more reqs of the current resolution
                return self._running_res

        sche_res = None
        for res in self.resolutions:
            req_ids = self.request_pool.get_unfinished_req_ids_by_res(res)
            if len(req_ids) > 0:
                sche_res = res
        self._running_res = sche_res
        return sche_res
    

    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Schedule requests for next iteration."""
        # 1. Pick a resolution to run
        # Also Update running resolution if no reqs in this resolution
        res = self._choose_resolution()
        assert res is not None
    
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