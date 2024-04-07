import time

from typing import List, Tuple

from sduss.scheduler import Request, RequestStatus

from .policy import Policy


class FCFS(Policy):
    """First Come First Serve.
    
    FCFS always selects the oldest requests, and the find any other requests
    that can be batched with it (A giant request will be split as many single
    requests at the entrypoint. They can be processed together).
    """
    def get_priority(self, now: float, req: Request) -> float:
        """Returns a float as a comparison metric for sorted.

        Args:
            now (float): Time at now.
            req (Request): Requst.

        Returns:
            float: Comparison result.
        """        
        return now - req.arrival_time


    def decide_stage(self) -> RequestStatus:
        for 
        
        
        queue_list = list(state_queues.values())
        # Sort with descending order
        queue_list = sorted(queue_list, key=lambda x: x[1], reverse=True)
        for i in range(len(queue_list)):
            if len(queue_list[i][0]) > 0:
                return queue_list[i][2]
        return None
    
    
    def _flatten_all_reqs(self) -> List[Request]:
        reqs = []
        for status_queues in self.request_pool.values():
            for request_queue in status_queues.values():
                for req in request_queue.values():
                    reqs.append(req)
        return req
    
    
    def schedule_requests(self) -> Tuple[RequestStatus, List[Request]]:
        flattened_reqs = self._flatten_all_reqs()

        # Find the oldest request
        now = time.monotonic()
        flattened_reqs.sort(key = lambda req: now - req.arrival_time, reverse=True)
        target_req = flattened_reqs[0]

        # Find compatible requests
        # 1. has the same stage
        # 2. has the same remain steps
        # 3. sampling params is compatible

        
        
        pass
