import time

from typing import List, TYPE_CHECKING

from .policy import Policy
from ..wrappers import SchedulerOutput, WorkerReqStatus

if TYPE_CHECKING:
    from sduss.dispatcher import Request

class FCFS_Single(Policy):
    """First Come First Serve.
    
    FCFS always selects the oldest requests, and the find any other requests
    that can be batched with it. 

    FCFS features
        Supports:
            1. batch reqs of different timesteps
        Don't supports:
            2. mixed-precision shceduling
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
        target_res = target_req.sampling_params.resolution

        resolution_req_dict = {}
        
        # Find compatible requests
        # 1. has the same status
        compatible_reqs = self.request_pool.get_reqs_by_complex(status=target_status, resolution=target_res)
        # 2. sampling params is compatible
        num_to_collect = max_num
        for req in compatible_reqs:
            if num_to_collect <= 0:
                break
            if req.sampling_params.is_compatible_with(target_req.sampling_params): 
                resolution_req_dict[req.request_id] = req
                num_to_collect -= 1
        
        # wrapper
        ret = {}
        ret[target_res] = resolution_req_dict
    
        return SchedulerOutput(
            scheduled_requests=ret,
            status=target_status,
        )