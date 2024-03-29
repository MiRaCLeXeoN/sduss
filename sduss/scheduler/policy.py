from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

from sduss.scheduler.scheduler import StateQueue
from sduss.scheduler.wrappers import Request, RequestStatus

class Policy(ABC):
    
    @abstractmethod
    def get_priority(
        self,
        now: float,
        req: Request,
    ) -> float:
        raise NotImplementedError(
            "Policy base class method get_priority is called. Please implement"
            "this method in the derivative class for any usage."
        )

    
    @abstractmethod
    def decide_stage(
        self,
        state_queues: Dict[str, StateQueue]
    ) -> RequestStatus:
        raise NotImplementedError(
            "Policy bae calss method decide_stage is called. Please implement"
            "this method in the derivative for any usage."
        )

    
    def sort_by_priority(
        self,
        now: float,
        reqs: List[Request],
    ) -> List[Request]:
        return sorted(
            reqs,
            key=lambda req: self.get_priority(now, req),
            reverse=True,
        )


class FCFS(Policy):
    """First Come First Serve."""
    def get_priority(self, now: float, req: Request) -> float:
        """Returns a float as a comparison metric for sorted.

        Args:
            now (float): Time at now.
            req (Request): Requst.

        Returns:
            float: Comparison result.
        """        
        return now - req.arrival_time


    def decide_stage(self, state_queues: Dict[str, StateQueue]
    ) -> RequestStatus:
        queue_list = list(state_queues.values())
        # Sort with descending order
        queue_list = sorted(queue_list, key=lambda x: x[1], reverse=True)
        for i in range(len(queue_list)):
            if len(queue_list[i][0]) > 0:
                return queue_list[i][2]
        return None


class PolicyFactory:
    
    _POLICY_REGISTRY = {
        'fcfs': FCFS,
    }
    
    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)