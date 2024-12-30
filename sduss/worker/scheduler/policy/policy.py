from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..request_pool import WorkerRequestPool
    from ..wrappers import SchedulerOutput

class Policy(ABC):

    def __init__(self, **kwargs) -> None:
        # Reference scheduler's request pool
        self.request_pool : 'WorkerRequestPool' = kwargs.pop("request_pool")

    
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