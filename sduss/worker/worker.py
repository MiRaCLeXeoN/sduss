import os
import time

from typing import Optional, List, Dict, Union, TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from sduss.dispatcher import Request
from sduss.config import PipelineConfig, ParallelConfig, SchedulerConfig, EngineConfig
from sduss.model_executor import set_random_seed, get_pipeline_cls
from sduss.model_executor.parallel_utils.parallel_state import initialize_model_parallel
from sduss.logger import init_logger

from .scheduler import Scheduler
from .model_runner import ModelRunner
from .wrappers import WorkerOutput, WorkerRequest

if TYPE_CHECKING:
    from .wrappers import WorkerRequestDictType

logger = init_logger(__name__)

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
        engine_config: EngineConfig,
        rank: int,
        device: int,
        is_prepare_worker: bool = False,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        """
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
        self.engine_config = engine_config
        self.rank = rank
        self.device_num = device
        self.device = None
        self.is_prepare_worker = is_prepare_worker
        self.distributed_init_method = distributed_init_method

        self.use_esymred = pipeline_config.use_esymred
        self.use_mixed_precision = scheduler_config.use_mixed_precision

        # Updated and maintained by `execute_model` method
        self.request_pool: Dict[int, WorkerRequest] = {} 

        self.model_runner = ModelRunner(pipeline_config, parallel_config, scheduler_config, is_prepare_worker)
        self.scheduler: Scheduler = Scheduler()

        # compute local_rank, rank wrt current machine
        num_gpus = torch.cuda.device_count()
        if parallel_config.world_size <= num_gpus:
            self.local_rank = self.rank
        else:
            raise NotImplementedError("Cross-node distribution is not supported yet!")

        # global logger
        # if self.is_prepare_worker:
        #     logger = init_logger(__name__, to_file_name="./outputs/prepare_worker")
        # else:
        #     logger = init_logger(__name__, to_file_name="./outputs/gpu_worker")
        
    
    def init_dis_env(self) -> None:
        """Initialize model on designated device."""
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573

        # Check up
        assert self.is_prepare_worker == False

        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        # ? This env var set by Ray causes exceptions with graph building
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        
        self.device = torch.device(f"cuda:{self.device_num}")
        torch.cuda.set_device(self.device)

        _init_distributed_environment(self.parallel_config, self.rank, 
                                      self.distributed_init_method)
        
        set_random_seed(self.pipeline_config.seed)

        logger.debug(f"Worker rank={self.rank} local_rank={self.local_rank} complete initialization")
    
    
    def init_prepare(self) -> None:
        assert self.is_prepare_worker
        # rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
        # local_rank = int(os.getenv("LOCAL_RANK"))
        # logger.debug(f"rank={self.rank}, local_rank={local_rank}")
        set_random_seed(self.pipeline_config.seed)
    

    def load_model(self):
        self.model_runner.load_model()
        self.scheduler = Dispatcher(self.scheduler_config, self.parallel_config, 
                                   self.engine_config, self.model_runner.pipeline.SUPPORT_RESOLUTIONS)
                                
    
    def step(self) -> Optional[WorkerOutput]:
        pass
    

    def add_requests(self, req_ids: List[int], req_sps: List[Any]):
        pass
    
    
    def remove_requests_by_id(self, req_ids: Union[int, List[int]]):
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        for req_id in req_ids:
            del self.request_pool[req_id]
    
    
    def receive_prepare_output(self, prepare_output: WorkerOutput):
        """Receive prepare output from engine."""
        start_time = time.time()
        self._process_prepare_output(prepare_output=prepare_output)
        end_time = time.time()
        return WorkerOutput(
            start_time=start_time,
            end_time=end_time
        )
        
    
    def exec_prepare_stage(
        self,
        scheduler_reqs: List[Request],
        use_mixed_precision: bool,
        prepare_output: WorkerOutput = None,
    ) -> None:
        """Execute prepare stage inference.
        
        At this stage, mixed precision doesn't matter at all.

        Args:
            scheduler_reqs (List[Request]): _description_
        """
        # 0. Store prepare results
        if prepare_output is not None:
            self._process_prepare_output(prepare_output)

        # 1. Create WorkerRequests to track reqs.
        worker_reqs: "WorkerRequestDictType" = {}
        for sche_req in scheduler_reqs:
            wr = WorkerRequest(sche_req)
            # Only register when prepare stage is not overlapped
            if not self.is_prepare_worker:
                self.request_pool[wr.request_id] = wr
            
            res = wr.sampling_params.resolution
            if res not in worker_reqs:
                worker_reqs[res] = [wr]
            else:
                worker_reqs[res].append(wr)
        
        # 2. Execute
        start_time = time.time()
        self.model_runner.exec_prepare_stage(worker_reqs)
        end_time = time.time()

        # 3. Create return wrapper
        return WorkerOutput(
            worker_reqs=worker_reqs, 
            status=ReqStatus.PREPARE,
            start_time=start_time,
            end_time=end_time,
            is_from_prepare_worker=self.is_prepare_worker,
        )
        
    
    def exec_denoising_stage(
        self,
        req_ids: List[int],
        use_mixed_precision: bool,
        is_sliced: bool,
        patch_size: int,
        prepare_output: WorkerOutput = None,
    ):
        """Execute denoising stage.

        Requests that finishes denoising stage don't need to be returned. Scheduelr
        can track requests' status according to its data duplicates.

        Args:
            req_ids (List[int]): IDs of requests to execute one iteration.
        """
        # 0. Store prepare results
        if prepare_output is not None:
            self._process_prepare_output(prepare_output)
        
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
        start_time = time.time()
        self.model_runner.exec_denoising_stage(worker_reqs, is_sliced, patch_size)
        end_time = time.time()

        return WorkerOutput(
            start_time=start_time,
            end_time=end_time,
        )
    
    
    def exec_post_stage(
        self,
        req_ids: List[int],
        use_mixed_precision: bool,
        prepare_output: WorkerOutput = None,
    ) -> WorkerOutput:
        # 0. Store prepare results
        if prepare_output is not None:
            self._process_prepare_output(prepare_output)

        # 1. Collect requests and wrap as dict
        worker_reqs_dict: "WorkerRequestDictType" = {}
        for req_id in req_ids:
            wq = self.request_pool[req_id]
            res = wq.sampling_params.resolution
            if res not in worker_reqs_dict:
                worker_reqs_dict[res] = [wq]
            else:
                worker_reqs_dict[res].append(wq)

        start_time = time.time()
        self.model_runner.exec_post_stage(worker_reqs_dict)
        end_time = time.time()

        # Create output
        output = WorkerOutput(
            worker_reqs=worker_reqs_dict, 
            status=ReqStatus.POSTPROCESSING,
            start_time=start_time,
            end_time=end_time,
        )

        # Remove finished requests
        self.remove_requests_by_id(req_ids)

        return output


    def _process_prepare_output(self, prepare_output: WorkerOutput) -> None:
        """Process prepare output.
        
        Register worker requests from prepare stage.

        Args:
            prepare_output (WorkerOutput): Output.
        """
        # now = time.time()
        worker_reqs = prepare_output.worker_reqs
        for worker_req_list in worker_reqs.values():
            for wr in worker_req_list:
                self.add_request(wr.request_id, wr)
                wr.to_tensor()
                # Move tensors to current device
                wr.to_device(self.device)
                wr.to_dtype(self.pipeline_config.kwargs["torch_dtype"])
        # logger.info(f"Unpack overlapped prepare output: {time.time() - now}")

        
    def warm_up_model(self) -> None:
        """Capture the model and set seeds"""
        if not self.pipeline_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.pipeline_config.seed)
    
    
    def clear(self) -> None:
        dist.destroy_process_group()

    
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
    # tensor = torch.ones(1) * rank
    # tensor = tensor.cuda()
    # print(f"[Rank {rank}, device {torch.cuda.current_device()}] Before allreduce: {tensor.item()}")
    # torch.distributed.all_reduce(tensor)
    # print(f"[Rank {rank}, device {torch.cuda.current_device()}] After allreduce: {tensor.item()}")
    # initialize_model_parallel(parallel_config.tensor_parallel_size,
    #                           parallel_config.pipeline_parallel_size)