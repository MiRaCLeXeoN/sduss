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
    
    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Schedule requests for next iteration.

        Args:
            max_num (int): _description_

        Returns:
            List[Request]: _description_
        """
        flattened_reqs = self.request_pool.get_unfinished_reqs()

        # Find the oldest request
        now = time.time()
        flattened_reqs.sort(key = lambda req: now - req.arrival_time, reverse=True)
        target_req = flattened_reqs[0]
        target_status = target_req.status

        reqs_same_status = self.request_pool.get_reqs_by_complex(status=target_status)
        reqs_same_status.sort(key=lambda req: now - req.arrival_time, reverse=True)

        res_reqs_dict: Dict[int, Dict[int, Request]] = {}
        
        # Collect reqs
        num_to_collect = max_num
        while num_to_collect > 0 and reqs_same_status:
            req = reqs_same_status.pop(0)
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
        is_sliced = True
        patch_size = 256
        
        return SchedulerOutput(
            scheduled_requests=res_reqs_dict,
            status=target_status,
            is_sliced=is_sliced,
            patch_size=patch_size,
        )