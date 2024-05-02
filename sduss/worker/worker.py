import os
from typing import Optional, List, Dict, Union, TYPE_CHECKING

import torch
import torch.distributed

from .model_runner import ModelRunner
from .wrappers import WorkerOutput, WorkerRequest
from sduss.config import PipelineConfig, ParallelConfig, SchedulerConfig

from sduss.scheduler import Request, RequestStatus
from sduss.model_executor import set_random_seed
from sduss.model_executor.parallel_utils.parallel_state import initialize_model_parallel

if TYPE_CHECKING:
    from .wrappers import WorkerRequestDictType

class Worker:
    """A worker GPU class
    
    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of distributed
    inference, each worker is assigned a partition of the model
    """
    
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        """ FIXME

        Args:
            model_config (ModelConfig): Model config
            parallel_config (ParallelConfig): Parallel config
            scheduler_config (SchedulerConfig): Scheduler config
            rank (Optional[int], optional): _description_. Defaults to None.
            distributed_init_method (Optional[str], optional): _description_. Defaults to None.
        """
        self.pipeline_config = pipeline_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        self.use_esymred = pipeline_config.use_esymred
        self.use_mixed_precision = scheduler_config.use_mixed_precision

        # Updated and maintained by `execute_model` method
        self.request_pool: Dict[int, WorkerRequest] = {} 

        self.model_runner = ModelRunner(pipeline_config, parallel_config, scheduler_config)
        
    
    def init_dis_env(self) -> None:
        """Initialize model on designated device."""
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # ? This env var set by Ray causes exceptions with graph building
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        
        # Env vars will be set by Ray
        # ? What are these variables?
        self.rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank")
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        
        _init_distributed_environment(self.parallel_config, self.rank, 
                                      self.distributed_init_method)
        
        set_random_seed(self.pipeline_config.seed)
    

    def load_model(self):
        self.model_runner.load_model()
    

    def add_request(self, req_id: int, wq: WorkerRequest):
        self.request_pool[req_id] = wq
    
    
    def remove_requests(self, req_ids: Union[int, List[int]]):
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        for req_id in req_ids:
            del self.request_pool[req_id]
        
    
    def exec_prepare_stage(
        self,
        scheduler_reqs: List[Request],
        use_mixed_precision: bool,
    ) -> None:
        """Execute prepare stage inference.
        
        At this stage, mixed precision doesn't matter at all.

        Args:
            scheduler_reqs (List[Request]): _description_
        """
        # 1. Create WorkerRequests to track reqs.
        worker_reqs: "WorkerRequestDictType" = {}
        for sche_req in scheduler_reqs:
            wq = WorkerRequest(sche_req)
            self.add_request(sche_req.request_id, wq)
            res = wq.sampling_params.resolution
            if res not in worker_reqs:
                worker_reqs[res] = [wq]
            else:
                worker_reqs[res].append(wq)
        
        # 2. Execute
        self.model_runner.exec_prepare_stage(worker_reqs)

        # 3. Create return wrapper
        return WorkerOutput(worker_reqs=worker_reqs, status=RequestStatus.PREPARE)
        
    
    def exec_denoising_stage(
        self,
        req_ids: List[int],
        use_mixed_precision: bool,
        is_sliced: bool,
        patch_size: int,
    ):
        """Execute denoising stage.

        Requests that finishes denoising stage don't need to be returned. Scheduelr
        can track requests' status according to its data duplicates.

        Args:
            req_ids (List[int]): IDs of requests to execute one iteration.
        """
        # 1. Collect requests and wrap as dict
        worker_reqs: "WorkerRequestDictType" = {}
        for req_id in req_ids:
            wq = self.request_pool[req_id]
            res = wq.sampling_params.resolution
            if res not in worker_reqs:
                worker_reqs[res] = [wq]
            else:
                worker_reqs[res].append(wq)

        # 2. Execute
        self.model_runner.exec_denoising_stage(worker_reqs, is_sliced, patch_size)

        # 3. Update reqs states
        # But maybe unnecessary
        return
    
    
    def exec_post_stage(
        self,
        req_ids: List[int],
        use_mixed_precision: bool,
    ) -> WorkerOutput:
        # 1. Collect requests and wrap as dict
        worker_reqs_dict: "WorkerRequestDictType" = {}
        for req_id in req_ids:
            wq = self.request_pool[req_id]
            res = wq.sampling_params.resolution
            if res not in worker_reqs_dict:
                worker_reqs_dict[res] = [wq]
            else:
                worker_reqs_dict[res].append(wq)

        self.model_runner.exec_post_stage(worker_reqs_dict)

        # Create output
        output = WorkerOutput(worker_reqs=worker_reqs_dict, status=RequestStatus.POSTPROCESSING)

        # Remove finished requests
        self.remove_requests(req_ids)

        return output

        
    def warm_up_model(self) -> None:
        """Capture the model and set seeds"""
        if not self.pipeline_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.pipeline_config.seed)

    
def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """"Initialize the distributed environment"""
    
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size})."
            )
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized"
        )
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method
        )
    
    # warmup
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)