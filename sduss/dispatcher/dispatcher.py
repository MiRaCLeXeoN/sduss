import enum
import time
import sys
import pandas as pd

from typing import List, Optional, Tuple, Dict, Union, Iterable
from typing import TYPE_CHECKING

from sduss.config import SchedulerConfig, ParallelConfig, EngineConfig
from sduss.logger import init_logger

from .wrappers import Request, ReqStatus
from .request_pool import RequestPool
from .policy import DispatchPolicyFactory
from .utils import DispatcherResultType

logger = init_logger(__name__)

class Dispatcher:
    """Dispatcher to distribute requests across workers."""
    
    def __init__(
        self,
        dispatcher_config: SchedulerConfig,
        parallel_config: ParallelConfig,
        engine_config: EngineConfig,
        support_resolutions: List[int],
    ) -> None:
        self.scheduler_config = dispatcher_config
        self.parallel_config = parallel_config
        self.engine_config = engine_config
        self.support_resolutions = support_resolutions

        self.use_mixed_precision = dispatcher_config.use_mixed_precision
        self.dp_size = self.parallel_config.data_parallel_size

        # Request pool manages all requests
        self.request_pool = RequestPool(self.dp_size)

        # Scheduler policy
        self.policy = DispatchPolicyFactory.get_policy(policy_name=self.scheduler_config.policy,
                                                        use_mixed_precision=self.use_mixed_precision,
                                                        request_pool=self.request_pool,
                                                        dp_size=self.dp_size)
        
        # Used for logging
        self.cycle_counter = 0
        

    def add_requests(self, reqs: Union[List[Request], Request]) -> None:
        """Add a new request to waiting queue."""
        if not isinstance(reqs, list):
            reqs = [reqs]
        
        self.request_pool.add_requests(reqs)
        return None
        

    def abort_requests(self, request_ids: Union[int, Iterable[int]]) -> List[Request]:
        """Abort a handful of requests.

        Args:
            request_ids (Union[str, Iterable[str]]): Requests to be aborted.
        """
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        
        aborted_reqs = self.request_pool.remove_requests(request_ids)
        return aborted_reqs
        
    
    def dispatch(self) -> DispatcherResultType:
        dispatched_reqs = self.policy.dispatch_requests()
        self.request_pool.update_requests(sum(dispatched_reqs.values(), []))
        return dispatched_reqs
    
    
    def log_status(self):
        pass