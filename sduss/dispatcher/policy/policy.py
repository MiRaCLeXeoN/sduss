from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sduss.dispatcher.wrappers import ResolutionRequestQueue, SchedulerOutput

class DispatchPolicy(ABC):

    def __init__(self, **kwargs) -> None:
        # Reference scheduler's request pool
        # FIXME:
        pass
    
    @abstractmethod
    def add_request(self, req) -> None:
        raise NotImplementedError("You must implemente this method in the derived class.")