from typing import List, Dict
from collections import defaultdict

from .policy import DispatchPolicy
from ..wrappers import Request, ReqStatus

class GreedyDispath(DispatchPolicy):
    """Greedy method to dispatch requests to workers.
        Requests are added to the worker with minimal workload in the next iteration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

    def dispatch_requests(self) -> Dict[int, List[Request]]:
        # 1. Get all waiting reqs
        _waiting_req_ids = self.request_pool.get_ids_by_status(ReqStatus.WAITING)
        waiting_reqs = self.request_pool.get_by_ids(_waiting_req_ids)
        # 2. Get the workload all ranks
        pixels_by_dp_rank = self.request_pool.get_pixels_all_dp_rank()
        res = defaultdict(list)
        # 3. dispatch reqs to the rank with minimal workload
        for req in waiting_reqs:
            # 3.1 get the rank with minimal workload
            target_dp_rank = min(pixels_by_dp_rank, key=pixels_by_dp_rank.get)
            # 3.2 update req attributes
            req.status = ReqStatus.RUNNING
            req.dp_rank = target_dp_rank
            # 3.3 update dict
            pixels_by_dp_rank[target_dp_rank] += req.sampling_params.resolution ** 2

            res[target_dp_rank].append(req)

        return res