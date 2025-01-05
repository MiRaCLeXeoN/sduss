import enum
import time
import sys
import pandas as pd

from typing import List, Optional, Tuple, Dict, Union, Iterable, TYPE_CHECKING

from sduss.config import SchedulerConfig, ParallelConfig, EngineConfig
from sduss.logger import init_logger

from .wrappers import Request, ReqStatus
from .request_pool import RequestPool
from .policy import DispatchPolicyFactory
from .utils import DispatcherResultType

if TYPE_CHECKING:
    from sduss.worker.wrappers import WorkerOutput

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
        self.policy = DispatchPolicyFactory.get_policy(policy_name=self.engine_config.dispatcher_policy,
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
        
        if len(request_ids) == 0:
            return []
        
        aborted_reqs = self.request_pool.remove_requests(request_ids)

        finish_time = time.time(0)
        for req in aborted_reqs:
            req.output = None
            req.status = ReqStatus.ABORTED
            req.finish_time = finish_time

        return aborted_reqs
        
    
    def dispatch(self) -> DispatcherResultType:
        # ! Attributes of the reqeusts should be modified by the call to `dispatch_requests`
        dispatched_reqs = self.policy.dispatch_requests()
        if len(dispatched_reqs) > 0:
            self.request_pool.update_requests(sum(dispatched_reqs.values(), []))
        return dispatched_reqs
    
    
    def process_worker_outputs(self, worker_outputs: 'List[WorkerOutput]') -> List[Request]:
        # 1. flatten
        req_ids = []
        outputs = []
        abort_req_ids = []
        for wo in worker_outputs:
            abort_req_ids.extend(wo.aborted_reqs)
            for req_id, output in wo.req_output_dict.items():
                req_ids.append(req_id)
                outputs.append(output)
        
        # 2. Get reqs, assign output, update status
        finished_reqs = self.request_pool.remove_requests(req_ids)
        finish_time = time.time()
        for req, output in zip(finished_reqs, outputs):
            req.output = output
            req.status = ReqStatus.FINISHED
            req.finish_time = finish_time
        
        # 3. Abort reqs
        aborted_reqs = self.abort_requests(abort_req_ids)
        
        return finished_reqs, aborted_reqs
    
    
    def has_unfinished_reqs(self) -> bool:
        return self.request_pool.has_unfinished_reqs()
    

    def get_num_unfinished_reqs_by_dp_rank(self) -> Dict:
        return self.request_pool.get_num_unfinished_reqs_by_dp_rank()
    
    def log_status(self):
        pass