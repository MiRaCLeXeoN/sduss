from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..request_pool import RequestPool, Request

class DispatchPolicy(ABC):

    def __init__(self, **kwargs) -> None:
        self.request_pool: 'RequestPool' = kwargs.pop("request_pool")
        self.dp_size = kwargs.pop("dp_size")
    

    @abstractmethod
    def dispatch_requests(self) -> 'Dict[int, List[Request]]':
        raise NotImplementedError("You must implemente this method in the derived class.")