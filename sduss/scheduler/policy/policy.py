from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

from sduss.scheduler.wrappers import Request, RequestStatus

class Policy(ABC):

    def __init__(self, request_pool: Dict[int, Dict[str, Dict[int, Request]]]) -> None:
        # Reference scheduler's request pool
        self.request_pool = request_pool
    
    @abstractmethod
    def schedule_requests(self) -> List[Request]:
        raise NotImplementedError("You must implemente this method in the derived class.")

    
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
