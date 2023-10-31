
import os
from typing import Optional, List, Dict

import torch
import torch.distributed

from sduss.config import ModelConfig, ParallelConfig, SchedulerConfig
from sduss.sequence import SequenceGroupMetadata, SamplerOutput
from sduss.model_executor import get_model, set_random_seed
from sduss.model_executor.parallel_utils.parallel_state import initialize_model_parallel

class Worker:
    """A worker GPU class
    
    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of distributed
    inference, each worker is assigned a partition of the model
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
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
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.sliding_window = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None
        
    
    def init_model(self) -> None:
        """Initialize model on designated device"""
        # ? This env var set by Ray causes exceptions with graph building
        # os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        
        # Env vars will be set by Ray
        # ? What are these variables?
        self.rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank")
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        
        _check_if_gpu_supports_dtype(self.model_config.dtype)
        
        _init_distributed_environment(self.parallel_config, self.rank, 
                                      self.distributed_init_method)
        
        set_random_seed(self.model_config.seed)
        self.model = get_model(self.model_config) # ? How to init distributed model?
        
    @torch.inference_mode()
    def profile_num_available_blocks(self) -> None:
        raise NotImplementedError("vllm parts not implemented yet")

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operation
        
        

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
        

def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype) -> None:
    """Check if the GPU supports the dtype.
    
    Only `torch.bfloat16` is checked for GPUs with capability under 8.x.
    """
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}.")