import time

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sduss.scheduler import Request, RequestStatus, SchedulerOutput

from .policy import Policy


class FCFS(Policy):
    """First Come First Serve.
    
    FCFS always selects the oldest requests, and the find any other requests
    that can be batched with it (A giant request will be split as many single
    requests at the entrypoint. They can be processed together).
    """
    def _flatten_all_reqs(self) -> List[Request]:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_reqs())
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
        now = time.monotonic()
        flattened_reqs.sort(key = lambda req: now - req.arrival_time, reverse=True)
        target_req = flattened_reqs[0]
        target_status = target_req.status
        target_res = target_req.sampling_params.resolution

        resolution_req_dict = {}
        
        # Find compatible requests
        # 1. has the same status
        res_queue = self.request_pool[target_res]
        queue = res_queue.get_queue_by_status(target_status)
        # 2. sampling params is compatible
        num_to_collect = max_num
        for req in queue.values():
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