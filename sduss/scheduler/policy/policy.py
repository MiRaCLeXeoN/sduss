from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sduss.scheduler.wrappers import ResolutionRequestQueue, SchedulerOutput

class Policy(ABC):

    def __init__(self, **kwargs) -> None:
        # Reference scheduler's request pool
        self.request_pool : Dict[int, 'ResolutionRequestQueue'] = kwargs.pop("request_pool")
        self.non_blocking_step: bool = kwargs.pop("non_blocking_step")
        self.overlap_prepare: bool = kwargs.pop("overlap_prepare")

    
    @abstractmethod
    def schedule_requests(self, max_num: int) -> 'SchedulerOutput':
        """Schedule requests for next iteration.

        This method should be overwritten by derived policies.

        Args:
            max_num (int): Number of requests to be scheduled at maximum.

        Returns:
            SchedulerOutput: output
        """
        raise NotImplementedError("You must implemente this method in the derived class.")
    
    
    @abstractmethod
    def scheduler_request_overlap_prepare(
        self, 
        max_num: int, 
        max_overlapped_prepare_reqs: int,
        accept_overlap_prepare_reqs: bool,
    ) -> 'SchedulerOutput':
        """Scheduler requests for next iteration regarding overlapping prepare
        stage requests.

        Args:
            max_num (int): Maximum number of requests to scheduler within each round.
            max_overlapped_prepare_reqs (int): Maximum number of requests to scheduler within
                each round.

        Returns:
            SchedulerOutput: Output.
        """
        raise NotImplementedError("You must implemente this method in the derived class.")