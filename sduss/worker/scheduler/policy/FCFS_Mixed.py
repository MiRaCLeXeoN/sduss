import time

from typing import List, TYPE_CHECKING, Dict

from .policy import Policy
from ..wrappers import SchedulerOutput, WorkerReqStatus
from ..utils import find_gcd, convert_list_to_res_dict


if TYPE_CHECKING:
    from sduss.dispatcher import Request

class FCFS_Mixed(Policy):
    """First Come First Serve.
    
    FCFS always selects the oldest requests.
    FCFS features
        Supports:
            1. Dynamic Batching
            2. Support mixed precision
        Doesn't support:
            1.
    """
    def _flatten_all_reqs(self) -> List['Request']:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_unfinished_normal_reqs())
        return reqs
    
    
    def _get_all_reqs_by_status(self, status: "WorkerReqStatus") -> List['Request']:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_reqs_by_status(status))
        return reqs
    
    
    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Schedule requests for next iteration.

        Args:
            max_num (int): _description_

        Returns:
            List[Request]: _description_
        """
        flattened_reqs = self._flatten_all_reqs()

        if len(flattened_reqs) == 0:
            # No reqs to schedule
            return SchedulerOutput(
                scheduled_requests={},
                status=WorkerReqStatus.EMPTY,
            )

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
        if target_status == WorkerReqStatus.DENOISING:
            if len(res_reqs_dict) > 1:
                is_sliced = True
                patch_size = find_gcd(list(res_reqs_dict))
            else:
                is_sliced = False
                patch_size = list(res_reqs_dict.keys())[0]
        
        return SchedulerOutput(
            scheduled_requests=res_reqs_dict,
            status=target_status,
            is_sliced=is_sliced,
            patch_size=patch_size,
        )