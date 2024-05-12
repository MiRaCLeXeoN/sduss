import enum
import time

from typing import List, Optional, Tuple, Dict, Union, Iterable
from typing import TYPE_CHECKING

from sduss.config import SchedulerConfig
from sduss.logger import init_logger

from .policy import PolicyFactory
from .wrappers import Request, RequestStatus, SchedulerOutput, ResolutionRequestQueue

if TYPE_CHECKING:
    from .wrappers import SchedulerOutputReqsType
    from sduss.worker import WorkerOutput

logger = init_logger(__name__)

class Scheduler:
    """Main scheduler which arranges tasks.
    
    Attributes:
        prompt_limit: Length limit of the prompt derived from configuration.
    """
    
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.scheduler_config = scheduler_config

        # Unpack scheduler config's argumnents
        self.max_batchsize = scheduler_config.max_batchsize
        self.use_mixed_precision = scheduler_config.use_mixed_precision
        
        # resolution -> queues -> RequestQueue
        self.request_pool: Dict[int, ResolutionRequestQueue] = {}
        # req_id -> req, for fast reference
        self.req_mapping: Dict[int, Request] = {}
        # Lazy import to avoid circular import
        from sduss.scheduler import SUPPORT_RESOLUTION
        for res in SUPPORT_RESOLUTION:
            self._initialize_resolution_queues(res)

        # Scheduler policy
        self.policy = PolicyFactory.get_policy(policy_name=self.scheduler_config.policy,
                                               request_pool=self.request_pool,
                                               use_mixed_precision=self.use_mixed_precision)
        
        # Logs
        self.cycle_counter = 0
        

    def schedule(self) -> SchedulerOutput:
        """Schedule requests for next iteration."""
        scheduler_output = self.policy.schedule_requests(max_num=self.max_batchsize)
        # More wrappers will be added here.

        # FIXME: DEBUG
        self.cycle_counter += 1
        
        return scheduler_output
    
    
    def schedule_overlap_prepare(self) -> SchedulerOutput:
        """Scheduler requests with overlapped prepare stage."""
        scheduler_output = self.policy.scheduler_request_overlap_prepare(max_num=self.max_batchsize)
        # More wrappers will be added here.

        # FIXME: DEBUG
        self.cycle_counter += 1
        
        return scheduler_output
        
        
    def add_request(self, req: Request) -> None:
        """Add a new request to waiting queue."""
        resolution = req.sampling_params.resolution
        self.request_pool[resolution].add_request(req)
        assert req.request_id not in self.req_mapping
        self.req_mapping[req.request_id] = req
        

    def abort_request(self, request_ids: Union[int, Iterable[int]]) -> None:
        """Abort a handful of requests.

        Args:
            request_ids (Union[str, Iterable[str]]): Requests to be aborted.
        """
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        res_reqid_dict: Dict[int, List[int]] = {}
        for req_id in request_ids:
            req = self.req_mapping.pop(req_id)  # pop out mapping
            resolution = req.sampling_params.resolution
            if resolution not in res_reqid_dict:
                res_reqid_dict[resolution] = [req_id]
            else:
                res_reqid_dict[resolution].append(req_id)
        
        # Abort reqs in resolution queues
        for res in res_reqid_dict:
            self.request_pool[res].abort_requests(res_reqid_dict[res])
        return
        
    
    def has_unfinished_requests(self) -> bool:
        for res_queue in self.request_pool.values():
            if res_queue.get_num_unfinished_reqs() > 0:
                return True
        return False
        

    def get_num_unfinished_requests(self) -> int:
        total = 0
        for res_queue in self.request_pool.values():
            total += res_queue.get_num_unfinished_reqs()
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
        scheduler_outputs: SchedulerOutput,
        output: "WorkerOutput",
        req_ids: List[int],
    ):
        """Update requests after one iteration."""
        # Move reqs to next status
        sche_status = scheduler_outputs.status
        sche_reqs: "SchedulerOutputReqsType" = scheduler_outputs.scheduled_requests

        next_status = self._get_next_status(sche_status)
        if sche_status == RequestStatus.WAITING:
            self._update_reqs_to_next_status(prev_status=sche_status, next_status=next_status, reqs=sche_reqs)
            return 
        elif sche_status == RequestStatus.PREPARE:
            self._update_reqs_to_next_status(prev_status=sche_status, next_status=next_status, reqs=sche_reqs)
            # The real remaining step may not be initial parameter
            self._update_remain_steps(reqs=sche_reqs, reqs_steps_dict=output.reqs_steps_dict)
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
            raise RuntimeError(f"Unexpected status {sche_status} to update.")
    
    
    def update_reqs_status_nonblocking(
        self,
        scheduler_outputs: SchedulerOutput,
        req_ids: List[int],
        prepare_output: WorkerOutput,
        denoising_output,
        postprocessing_output: WorkerOutput,
        prev_scheduler_output: SchedulerOutput,
    ) -> List[Request]:
        """Update requests status regarding nonblocking execution paradigm.

        Args:
            scheduler_outputs (SchedulerOutput): Scheduler output in this round.
            req_ids (List[int]): Request ids in this round.
            prepare_output (WorkerOutput): Prepare output from previous round.
            denoising_output (None): Denoising output from previous round.
            postprocessing_output (WorkerOutput): Postprocessing output from previous round.
            prev_scheduler_output (SchedulerOutput): Scheduler output in previous round.

        Returns:
            List[Request]: Requests that can be freed.
        """
        # 0. If prepare_output available, use it to update requests
        if prepare_output is not None:
            self._update_remain_steps(reqs_steps_dict=prepare_output.reqs_steps_dict)

        finished_reqs: List[Request] = []
        # 1. Process output from previous round.
        prev_sche_status = prev_scheduler_output.status
        if prev_sche_status == RequestStatus.WAITING:
            pass
        elif prev_sche_status == RequestStatus.PREPARE:
            # update remain steps is done at step 0, nothing more to do here.
            pass
        elif prev_sche_status == RequestStatus.DENOISING:
            pass
        elif prev_sche_status == RequestStatus.POSTPROCESSING:
            assert postprocessing_output is not None
            # create finish timestamp
            current_timestamp = time.time()
            # store output
            for req_id in req_ids:
                self.req_mapping[req_id].output = postprocessing_output.req_output_dict[req_id]
                self.req_mapping[req_id].finish_time = current_timestamp
                finished_reqs.append(self.req_mapping[req_id])
        
        # 2. To ensure consistency, reqs in this round must be updated.
        sche_status = scheduler_outputs.status
        sche_reqs: "SchedulerOutputReqsType" = scheduler_outputs.scheduled_requests
        next_status = self._get_next_status(sche_status)
        if sche_status == RequestStatus.WAITING:
            self._update_reqs_to_next_status(prev_status=sche_status, next_status=next_status, reqs=sche_reqs)
        elif sche_status == RequestStatus.PREPARE:
            self._update_reqs_to_next_status(prev_status=sche_status, next_status=next_status, reqs=sche_reqs)
        elif sche_status == RequestStatus.DENOISING:
            # More steps done
            # Some reqs may need move to post stage
            denoising_complete_reqs = self._decrease_one_step(sche_reqs)
            if len(denoising_complete_reqs) > 0:
                self._update_reqs_to_next_status(prev_status=sche_status, 
                                                  next_status=RequestStatus.POSTPROCESSING, 
                                                  reqs=denoising_complete_reqs)
        elif sche_status == RequestStatus.POSTPROCESSING:
            # finished reqs should be freed by calls to `free_finished_reqs`
            self._update_reqs_to_next_status(prev_status=sche_status, next_status=next_status, reqs=sche_reqs)
        else:
            raise RuntimeError(f"Unexpected status {sche_status} to update.")
        
        return finished_reqs
            
        
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
        for req in reqs:
            res = req.sampling_params.resolution
            self.request_pool[res].free_finished_reqs(req)
        
        
    def _initialize_resolution_queues(self, res: int) -> None:
        self.request_pool[res] = ResolutionRequestQueue(res)

    
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