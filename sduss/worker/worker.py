
import os
from typing import Optional, List, Dict

import torch
import torch.distributed

from .model_runner import ModelRunner
from .wrappers import WorkerExecuteInput, WorkerOutput, WorkerRequest
from sduss.config import PipelineConfig, ParallelConfig, SchedulerConfig
from sduss.scheduler import Request, RequestStatus
from sduss.model_executor import get_pipeline, set_random_seed
from sduss.model_executor.parallel_utils.parallel_state import initialize_model_parallel

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
    
    
    def exec_prepare_stage(
        self,
        scheduler_reqs: List[Request],
    ) -> None:
        worker_reqs = []
        for sche_req in scheduler_reqs:
            wq = WorkerRequest(sche_req)
            self.request_pool[sche_req.request_id] = wq 
            worker_reqs.append(wq)
        
        self.model_runner.exec_prepare_stage(worker_reqs)
        
    
    def exec_denoising_stage(
        self,
        req_ids: List[int],
    ):
        """Execute denoising stage.

        Requests that finishes denoising stage don't need to be returned. Scheduelr
        can track requests' status according to its data duplicates.

        Args:
            req_ids (List[int]): IDs of requests to execute one iteration.
        """
        worker_reqs = []
        for req_id in req_ids:
            wq = self.request_pool[req_id]
            assert wq.step_input is not None
            worker_reqs.append(wq)

        # concat inputs






        
    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> None:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        
        # Execute a forward pass with dummy inputs to profile the memory
        # usage of the model
        self.model_runner.profile_run()
        
        # Calculate
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.pipeline_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) // cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        
        num_cpu_blocks = max(num_cpu_blocks, 0)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks


    def warm_up_model(self) -> None:
        """Capture the model and set seeds"""
        if not self.pipeline_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.pipeline_config.seed)

    
    @torch.inference_mode()
    def execute_model(
        self,

        
    ) -> SamplerOutput:
        # Issue cache operation
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None
        
        # Wait for cache operations to finish
        if cache_events is not None:
            for event in cache_events:
                event.wait()
        
        # If no input, return immediately
        if not seq_group_metadata_list:
            return {}
        
        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)
        return output
        

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