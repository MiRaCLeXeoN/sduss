import enum
import time
import sys

from typing import List, Optional, Tuple, Dict, Union, Iterable
from typing import TYPE_CHECKING

from sduss.config import SchedulerConfig, EngineConfig
from sduss.logger import init_logger

from .policy import PolicyFactory
from ..wrappers import WorkerRequest, WorkerReqStatus
from .wrappers import SchedulerOutput
from .request_pool import WorkerRequestPool

if TYPE_CHECKING:
    from .wrappers import SchedulerOutputReqsType
    from sduss.worker import WorkerOutput
    from ..runner.wrappers import RunnerOutput

logger = init_logger(__name__)

class Scheduler:
    """Main scheduler which arranges tasks.
    
    Attributes:
        prompt_limit: Length limit of the prompt derived from configuration.
    """
    
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        engine_config: EngineConfig,
        support_resolutions: List[int],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.engine_config = engine_config
        self.support_resolutions = support_resolutions

        # Unpack scheduler config's argumnents
        self.max_batchsize = scheduler_config.max_batchsize
        self.use_mixed_precision = scheduler_config.use_mixed_precision
        
        # resolution -> queues -> RequestQueue
        self.request_pool = WorkerRequestPool(support_resolutions)

        # Scheduler policy
        self.policy = PolicyFactory.get_policy(policy_name=self.scheduler_config.policy,
                                               request_pool=self.request_pool,
                                               use_mixed_precision=self.use_mixed_precision,
                                               support_resolutions=self.support_resolutions)
        
        # Logs
        self.cycle_counter = 0
        

    def schedule(self) -> SchedulerOutput:
        """Schedule requests for next iteration."""
        self.cycle_counter += 1
        scheduler_output = self.policy.schedule_requests(max_num=self.max_batchsize)
        return scheduler_output
    
    
    def add_requests(self, reqs: List[WorkerRequest]) -> None:
        """Add a new request to waiting queue."""
        # Add to request pool
        self.request_pool.add_requests(reqs)
        

    def abort_requests(self, request_ids: Union[int, Iterable[int]]) -> List[WorkerRequest]:
        """Abort a handful of requests.

        Args:
            request_ids (Union[str, Iterable[str]]): Requests to be aborted.
        """
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        
        aborted_reqs = self.request_pool.remove_requests(request_ids)
        for req in aborted_reqs:
            req.status = WorkerReqStatus.FINISHED_ABORTED
        return aborted_reqs


    def has_unfinished_requests(self) -> bool:
        return self.request_pool.has_unfinished_reqs()
    

    def process_output(
        self,
        prev_sche_output: SchedulerOutput,
        prev_output: 'RunnerOutput',
    ) -> List[WorkerRequest]:
        """Process the output of previous round. If any requests are finished,
        output will be placed inside accordingly.

        Finihsed reqs will be returned and automatically removed from request pool

        Args:
            prev_sche_output (SchedulerOutput): Scheduler output in previous round
            prev_output (RunnerOutput): Output of previous round

        Returns:
            List[WorkerRequest]: Requests that are finished.
        """
        finished_reqs: List[WorkerRequest] = []

        # 1. Process output from previous round.
        prev_sche_status = prev_sche_output.status
        prev_req_ids = prev_sche_output.get_req_ids()
        prev_reqs = self.request_pool.get_by_ids(prev_req_ids)
        if prev_sche_status == WorkerReqStatus.POSTPROCESSING:
            assert prev_output is not None
            # create finish timestamp
            current_timestamp = time.time()
            # store output
            for req in prev_reqs:
                req.output = prev_output.req_output_dict[req.request_id]
                req.finish_time = current_timestamp
            finished_reqs = prev_reqs
        else:
            pass

        # 2. Free finished reqs
        if len(finished_reqs) > 0:
            self.request_pool.remove_requests(prev_req_ids)

        return finished_reqs


    def update_reqs_status(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Update requests status regarding nonblocking execution paradigm.

        Args:
            scheduler_output (SchedulerOutput): Scheduler output in this round.
        """
        # To ensure consistency, reqs in this round must be updated.
        sche_status = scheduler_output.status
        if sche_status == WorkerReqStatus.EMPTY:
            # Nothing to update
            return 

        cur_reqs = self.request_pool.get_by_ids(scheduler_output.get_req_ids())
        next_status = self._get_next_status(sche_status)

        if sche_status == WorkerReqStatus.PREPARE:
            self._update_reqs_to_next_status(reqs=cur_reqs, next_status=next_status)
        elif sche_status == WorkerReqStatus.DENOISING:
            # More steps done
            # Some reqs may need move to post stage
            denoising_complete_reqs = self._decrease_one_step(cur_reqs)
            if len(denoising_complete_reqs) > 0:
                self._update_reqs_to_next_status(next_status=WorkerReqStatus.POSTPROCESSING, 
                                                  reqs=denoising_complete_reqs)
        elif sche_status == WorkerReqStatus.POSTPROCESSING:
            # finished reqs should be freed by calls to `free_finished_reqs`
            self._update_reqs_to_next_status(next_status=next_status, reqs=cur_reqs)
        else:
            raise RuntimeError(f"Unexpected status {str(sche_status)} to update.")
        
        self.request_pool.update_requests(cur_reqs)
        
        
    def free_finished_requests(self) -> List[WorkerRequest]:
        """Untrack input reqs if they are finished."""
        return self.request_pool.free_finished_reqs()

    
    def _get_next_status(self, prev_status: WorkerReqStatus) -> WorkerReqStatus:
        return WorkerReqStatus.get_next_status(prev_status)
        
    
    def _update_reqs_to_next_status(
        self, 
        reqs: List[WorkerRequest],
        next_status: WorkerReqStatus, 
    ):
        # Update resolution by resolution
        for req in reqs:
            req.status = next_status


    def _decrease_one_step(self, reqs: List[WorkerRequest]
    ) -> Optional["SchedulerOutputReqsType"]:
        """Decrease one remain step for requests."""
        # We should not alter the original one
        denoising_complete_reqs = []
        for req in reqs:
            assert req.status == WorkerReqStatus.DENOISING
            req.remain_steps -= 1
            if req.remain_steps == 0:
                denoising_complete_reqs.append(req)
        return denoising_complete_reqs
    
    
    def get_log_status_str(self):
        res = f"Scheduler cycle {self.cycle_counter}\n"
        res += self.request_pool.get_log_status_str()
        return res
    
    
    def log_status(self):
        logger.debug(f"Scheduler cycle {self.cycle_counter}")
        self.request_pool.log_status()