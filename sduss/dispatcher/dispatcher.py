import enum
import time
import sys

from typing import List, Optional, Tuple, Dict, Union, Iterable
from typing import TYPE_CHECKING

from sduss.config import SchedulerConfig, ParallelConfig, EngineConfig
from sduss.logger import init_logger

from .wrappers import Request, RequestStatus, SchedulerOutput, ResolutionRequestQueue
from .policy import DispatchPolicyFactory

if TYPE_CHECKING:
    from .wrappers import SchedulerOutputReqsType
    from sduss.worker import WorkerOutput

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

        # req_id -> req, for fast reference
        self.req_mapping: Dict[int, Request] = {}
        # dp_rank -> Dict[req_id -> req]
        self.reqs_by_dp: Dict[int, Dict[int, Request]] = {}
        for i in range(self.parallel_config.data_parallel_size):
            self.reqs_by_dp[i] = {}

        # Scheduler policy
        self.policy = DispatchPolicyFactory.get_policy(policy_name=self.scheduler_config.policy,
                                                        reqs_by_dp=self.reqs_by_dp,
                                                        use_mixed_precision=self.use_mixed_precision,)
        
        # Logs
        self.cycle_counter = 0
        

    def add_requests(self, reqs: Union[List[Request], Request]) -> None:
        """Add a new request to waiting queue."""
        if not isinstance(reqs, list):
            reqs = [reqs]
        
        for req in reqs:
            assert req.request_id not in self.req_mapping
            self.req_mapping[req.request_id] = req

        self.policy.add_request(reqs)
        

    def abort_requests(self, request_ids: Union[int, Iterable[int]]) -> List[Request]:
        """Abort a handful of requests.

        Args:
            request_ids (Union[str, Iterable[str]]): Requests to be aborted.
        """
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        
        aborted_reqs = []
        res_reqid_dict: Dict[int, List[int]] = {}
        for req_id in request_ids:
            req = self.req_mapping.pop(req_id)

            aborted_reqs.append(req)

            resolution = req.sampling_params.resolution
            if resolution not in res_reqid_dict:
                res_reqid_dict[resolution] = [req_id]
            else:
                res_reqid_dict[resolution].append(req_id)
        
        # Abort reqs in resolution queues
        for res in res_reqid_dict:
            self.request_pool[res].abort_requests(res_reqid_dict[res])

        return aborted_reqs
        
    
    def has_unfinished_normal_requests(self, is_nonblocking: bool) -> bool:
        if is_nonblocking:
            # We must wait all requests freed instead of finished, since
            # some requests are still executing.
            for res_queue in self.request_pool.values():
                if res_queue.get_num_unfreed_normal_reqs() > 0:
                    return True
            return False
            
        for res_queue in self.request_pool.values():
            if res_queue.get_num_unfinished_normal_reqs() > 0:
                return True
        return False
        

    def get_num_unfinished_normal_reqs(self) -> int:
        total = 0
        for res_queue in self.request_pool.values():
            total += res_queue.get_num_unfinished_normal_reqs()
        return total
    
    
    def has_finished_requests(self) -> bool:
        for res_queue in self.request_pool.values():
            if res_queue.get_num_finished_reqs() > 0:
                return True
        return False
    
    
    def get_finished_requests(self) -> List[Request]:
        ret = []
        for res in self.request_pool:
            ret.extend(self.request_pool[res].get_finished_reqs())
        return ret
    
    
    def update_reqs_status(
        self,
        scheduler_output: SchedulerOutput,
        output: "WorkerOutput",
        req_ids: List[int],
    ):
        """Update requests after one iteration."""
        # 0 If forced to update waiting reqs, update them
        if scheduler_output.update_all_waiting_reqs:
            self._update_all_waiting_reqs()

        sche_status = scheduler_output.status
        sche_reqs: "SchedulerOutputReqsType" = scheduler_output.scheduled_requests

        next_status = self._get_next_status(sche_status)
        if sche_status == RequestStatus.WAITING:
            if scheduler_output.update_all_waiting_reqs:
                # Don't update twice
                return 
            self._update_reqs_to_next_status(prev_status=sche_status, next_status=next_status, reqs=sche_reqs)
        elif sche_status == RequestStatus.PREPARE:
            self._update_reqs_to_next_status(prev_status=sche_status, next_status=next_status, reqs=sche_reqs)
            # The real remaining step may not be initial parameter
            self._update_remain_steps(reqs_steps_dict=output.reqs_steps_dict)
            return 
        elif sche_status == RequestStatus.DENOISING:
            # More steps done
            # may or may not move to post stage
            denoising_complete_reqs = self._decrease_one_step(sche_reqs)
            if len(denoising_complete_reqs) > 0:
                self._update_reqs_to_next_status(prev_status=sche_status, 
                                                  next_status=RequestStatus.POSTPROCESSING, 
                                                  reqs=denoising_complete_reqs)
            return 
        elif sche_status == RequestStatus.POSTPROCESSING:
            # finished reqs should be freed by calls to `free_finished_reqs`
            self._update_reqs_to_next_status(prev_status=sche_status, next_status=next_status, reqs=sche_reqs)
            # create finish timestamp
            current_timestamp = time.time()
            # store output
            for req_id in req_ids:
                self.req_mapping[req_id].output = output.req_output_dict[req_id]
                self.req_mapping[req_id].finish_time = current_timestamp
            return 
        else:
            raise RuntimeError(f"Unexpected status {str(sche_status)} to update.")
        

    def free_all_finished_requests(self) -> None:
        """Untrack all finished requests."""
        # 1. collect req_ids and free reqs
        req_ids = []
        for res_queue in self.request_pool.values():
            req_ids.extend(res_queue.get_finished_req_ids())
            res_queue.free_all_finished_reqs()
        # 2. clear mapping reference
        for req_id in req_ids:
            self.req_mapping.pop(req_id)
    
    
    def free_finished_requests(self, reqs: List[Request]) -> None:
        """Untrack input reqs if they are finished."""
        # Extract request ids
        res_req_ids = {}
        for req in reqs:
            res = req.sampling_params.resolution
            if res not in res_req_ids:
                res_req_ids[res] = [req.request_id]
            else:
                res_req_ids[res].append(req.request_id)
            self.req_mapping.pop(req.request_id)

        for res in res_req_ids:
            self.request_pool[res].free_finished_reqs(res_req_ids[res])
        
        
    def _initialize_resolution_queues(self, res: int) -> None:
        for i in range(self.data_parallel_size):
            self.request_pool[i][res] = ResolutionRequestQueue(res)

    
    def _get_next_status(self, prev_status: RequestStatus) -> RequestStatus:
        return RequestStatus.get_next_status(prev_status)
        
    
    def _update_reqs_to_next_status(
        self, 
        prev_status: RequestStatus, 
        next_status: RequestStatus, 
        reqs: "SchedulerOutputReqsType",
    ):
        # Update resolution by resolution
        for res, reqs_dict in reqs.items():
            self.request_pool[res].update_reqs_status(reqs_dict=reqs_dict, 
                                                      prev_status=prev_status, 
                                                      next_status=next_status)

                                                      
    def _update_all_waiting_reqs(self) -> None:
        """This method whill update all waiting reqs to prepare status."""
        for res_queue in self.request_pool.values():
            res_queue.update_all_waiting_reqs_to_prepare()


    def _decrease_one_step(self, reqs: "SchedulerOutputReqsType"
    ) -> Optional["SchedulerOutputReqsType"]:
        """Decrease one remain step for requests."""
        # We should not alter the original one
        denoising_complete_reqs: "SchedulerOutputReqsType" = {}
        for res, reqs_dict in reqs.items():
            for req_id, req in reqs_dict.items():
                # Prepare stage has been updated to denoising, it's safe to do so
                assert req.status == RequestStatus.DENOISING
                req.remain_steps -= 1
                if req.remain_steps == 0:
                    # add to complete reqs dict
                    if res not in denoising_complete_reqs:
                        denoising_complete_reqs[res] = {req_id : req}
                    else:
                        denoising_complete_reqs[res][req_id] = req
        return denoising_complete_reqs
    
    
    def _update_remain_steps(self, reqs_steps_dict: Dict[int, int]):
        for req_id, remain_steps in reqs_steps_dict.items():
            req = self.req_mapping[req_id]
            req.remain_steps = remain_steps
    
    
    def log_status(self):
        logger.debug(f"Scheduler cycle {self.cycle_counter}")
        for res in self.request_pool:
            resolution_queue = self.request_pool[res]
            logger.debug(resolution_queue.log_status(return_str=True))