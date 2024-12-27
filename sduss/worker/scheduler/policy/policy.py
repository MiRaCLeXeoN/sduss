from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sduss.dispatcher.wrappers import ResolutionRequestQueue, SchedulerOutput

class Policy(ABC):

    def __init__(self, **kwargs) -> None:
        # Reference scheduler's request pool
        self.request_pool : List[Dict[int, 'ResolutionRequestQueue']] = kwargs.pop("request_pool")

    
    @abstractmethod
    def add_request(self, req) -> None:
        raise NotImplementedError("You must implemente this method in the derived class.")

    
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