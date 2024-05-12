import time

from typing import List, TYPE_CHECKING, Dict

from .policy import Policy
from ..wrappers import SchedulerOutput, RequestStatus
from ..utils import find_gcd


if TYPE_CHECKING:
    from sduss.scheduler import Request

class FCFS_Mixed(Policy):
    """First Come First Serve.
    
    FCFS always selects the oldest requests.

    Support mixed precision.
    """
    def _flatten_all_reqs(self) -> List['Request']:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_unfinished_reqs())
        return reqs
    
    
    def _get_all_reqs_by_status(self, status: "RequestStatus") -> List['Request']:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_reqs_by_status(status))
        return reqs
    
    
    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Schedule requests for next iteration.

        FCFS features
            Supports:
                1. batch reqs of different timesteps
            Don't supports:
                2. mixed-precision shceduling

        Args:
            max_num (int): _description_

        Returns:
            List[Request]: _description_
        """
        flattened_reqs = self._flatten_all_reqs()

        # Find the oldest request
        now = time.time()
        flattened_reqs.sort(key = lambda req: now - req.arrival_time, reverse=True)
        target_req = flattened_reqs[0]
        target_status = target_req.status

        queue = self._get_all_reqs_by_status(target_status)
        queue.sort(key=lambda req: now - req.arrival_time, reverse=True)

        res_reqs_dict: Dict[int, Dict[int, Request]] = {}
        
        # Collect reqs
        num_to_collect = max_num
        while num_to_collect > 0 and queue:
            req = queue.pop(0)
            res = req.sampling_params.resolution
            if res not in res_reqs_dict:
                res_reqs_dict[res] = {req.request_id : req}
            else:
                res_reqs_dict[res][req.request_id] = req
            num_to_collect -= 1
        
        # Mixed precision arguments
        is_sliced = None
        patch_size = None
        # Only apply for denoising stage
        if target_status == RequestStatus.DENOISING:
            if len(res_reqs_dict) > 1:
                is_sliced = True
                patch_size = find_gcd(list(res_reqs_dict))
            else:
                is_sliced = False
                patch_size = list(res_reqs_dict.keys())[0]
        
        # FIXME: arrange prepare stage
        
        return SchedulerOutput(
            scheduled_requests=res_reqs_dict,
            status=target_status,
            is_sliced=is_sliced,
            patch_size=patch_size,
        )