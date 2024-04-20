import enum
import time

from typing import List, Optional, Tuple, Dict, Union, Iterable

from .policy import PolicyFactory
from .wrappers import Request, RequestStatus, SchedulerOutput, ResolutionRequestQueue
from sduss.config import SchedulerConfig
from sduss.logger import init_logger
from sduss.utils import Counter
from sduss.worker import WorkerOutput

logger = init_logger(__name__)

class PreemptionMode(enum.Enum):
    """Preemption Modes

    Attributes:
        SWAP: Swap out the blocks of the preempted sequences to CPU memory
            and swap the back in when the sequences are resumed.
        RECOMPUTE: Discard the blocks of the preempted sequences and recompute
            them when the sequences are resumed, treating the sequences as
            new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE  = enum.auto()


# req_id -> req
StateQueue = Tuple[List[Request], int, RequestStatus]
        
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
                                               request_pool=self.request_pool)
        
        # Set status mapping: name -> (priority, status)
        # Priority here only reflects the order of process stage
        self.status_mapping = {
            "waiting": (10, RequestStatus.WAITING),
            "prepare": (20, RequestStatus.PREPARE),
            "denoising": (30, RequestStatus.DENOISING),
            "postprocessing": (40, RequestStatus.POSTPROCESSING),
        }


    def schedule(self) -> SchedulerOutput:
        """Schedule requests for next iteration."""
        scheduler_output = self.policy.schedule_requests(self.max_batchsize)
        # More wrappers will be added here.
        
        return scheduler_output

        
    def add_request(self, req: Request) -> None:
        """Add a new request to waiting queue."""
        resolution = req.sampling_params.resolution
        self.request_pool[resolution].add_request(req)
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
            if not resolution in res_reqid_dict:
                res_reqid_dict[resolution] = [req_id]
            else:
                res_reqid_dict[resolution].append(req_id)
        
        # Abort reqs in resolution queues
        for res in res_reqid_dict:
            self.request_pool[res].abort_requests(res_reqid_dict[res])
        
        return
        
    
    def has_unfinished_requests(self) -> bool:
        for res_queue in self.request_pool.values():
            if res_queue.get_num_unfinished_reqs > 0:
                return True
        return False
        

    def get_num_unfinished_requests(self) -> int:
        total = 0
        for res_queue in self.request_pool.values():
            total += res_queue.get_num_unfinished_reqs
        return total
    
    
    def update_reqs_status(
        self,
        scheduler_outputs: SchedulerOutput,
        request_ids: List[int],
        output: WorkerOutput,
    ):
        """Update requests after one iteration."""
        # Move this req to next queue
        sche_status = scheduler_outputs.status
        sche_reqs = scheduler_outputs.scheduled_requests
        next_status = self._get_next_status(sche_status)
        if sche_status == RequestStatus.WAITING:
            self._update_reqs_to_status_queue(prev_status=sche_status, status=next_status, reqs=sche_reqs)
            return 
        elif sche_status == RequestStatus.PREPARE:
            # First iteration of denoising has done
            self._update_reqs_to_status_queue(prev_status=sche_status, status=next_status, reqs=sche_reqs)
            # Some reqs might only iterate once
            denoising_complete_reqs = self._decrease_one_step(sche_reqs)
            if len(denoising_complete_reqs) > 0:
                self._update_reqs_to_status_queue(prev_status=next_status, 
                                                  status=RequestStatus.POSTPROCESSING, 
                                                  reqs=denoising_complete_reqs)
            return 
        elif sche_status == RequestStatus.DENOISING:
            # More steps done
            # may or may not move to post stage
            denoising_complete_reqs = self._decrease_one_step(sche_reqs)
            if len(denoising_complete_reqs) > 0:
                self._update_reqs_to_status_queue(prev_status=next_status, 
                                                  status=RequestStatus.POSTPROCESSING, 
                                                  reqs=denoising_complete_reqs)
            return 
        elif sche_status == RequestStatus.POSTPROCESSING:
            assert output is not None
            # engine is responsible ofr prepare output, we don't need to do so here
            # free finished requests automatically
            self._free_finished_requests(sche_reqs)
            
        
    def _free_finished_requests(self, reqs: List[Request]) -> None:
        """Untrack all finished requests."""
        for req in reqs:
            assert req.status == RequestStatus.POSTPROCESSING
            self.postprocess.remove(req)
        
        for req in reqs:
            return 
            
        
    def _decide_stage(self) -> RequestStatus:
        target_status = self.policy.decide_stage(self.state_queue)
        if target_status is None:
            raise RuntimeError("No queue in scheduler has remaining requsts to process.")
        return target_status


    def _initialize_resolution_queues(self, res: int) -> None:
        self.request_pool[res] = ResolutionRequestQueue(res)

    
    def _get_next_status(self, prev_status: RequestStatus) -> RequestStatus:
        if prev_status == RequestStatus.WAITING:
            return RequestStatus.PREPARE
        elif prev_status == RequestStatus.PREPARE:
            return RequestStatus.DENOISING
        elif prev_status == RequestStatus.DENOISING:
            return RequestStatus.POSTPROCESSING
        elif prev_status == RequestStatus.POSTPROCESSING:
            return RequestStatus.FINISHED_STOPPED
        else:
            raise ValueError(f"Not next status available for {prev_status}")
    
    
    def _update_reqs_to_status_queue(self, prev_status: RequestStatus, status: RequestStatus, reqs: List[Request]):
        prev_queue = self._get_queue_from_status(prev_status)
        target_queue = self._get_queue_from_status(status)
        for req in reqs:
            prev_queue.remove(req)
            req.status = status
            target_queue.append(req)
 

    def _decrease_one_step(self, reqs: List[Request]) -> List[Request]:
        denoising_complete_reqs: List[Request] = []
        for req in reqs:
            # Prepare stage has been updated to denoising, it's safe to do so
            assert req.status == RequestStatus.DENOISING
            req.remain_steps -= 1
            if req.remain_steps == 0:
                denoising_complete_reqs.append(req)
        return denoising_complete_reqs