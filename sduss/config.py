"""All configuration classes """
from typing import Optional, Union, Dict

from sduss.logger import init_logger
from sduss.utils import get_cpu_memory
from sduss.utils import is_hip


logger = init_logger(__name__)

_GB = 1 << 30
class PipelineConfig:
    """Configuration for Pipeline.
    
    Pipelines are default to use diffusers' pipelines.

    Args:
        pipeline (str): Name or path of the huggingface pipeline to use.
        trust_remote_code (bool): Trust code from remote, e.g. from Huggingface, when
            downloading the model and tokenizer
        seed (int): _description_
        kwargs (Dict): kwargs for initializing moduels with `from_pretrained` method.
    """
    
    def __init__(
        self,
        pipeline: str,
        trust_remote_code: bool,
        seed: int,
        use_esymred: bool,
        use_batch_split: bool,
        kwargs: Dict,
    ) -> None:
        self.pipeline = pipeline
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.use_esymred = use_esymred
        self.use_batch_split = use_batch_split
        self.kwargs = kwargs

        self._verify_args()

    
    def _verify_args(self):
        if self.use_batch_split and not self.use_esymred:
            raise ValueError("Only esymred pipelines support batch split feature!")
    
    
    def verify_with_scheduler_config(
        self,
        scheduler_config: "SchedulerConfig",
    ):
        if scheduler_config.use_mixed_precision != self.use_esymred:
            raise ValueError("When using esymred pipelines, scheduler is forced to use mixed_precision.")
        
        

class ParallelConfig:
    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        data_parallel_size: int,
        num_cpus_extra_worker: int,
        worker_use_ray: bool,
        max_parallel_loading_workers: Optional[int] = None,
    ) -> None:
        """Configuration for the distributed execution.

        Args:
            pipeline_parallel_size: Number of pipeline parallel groups.
            tensor_parallel_size: Number of tensor parallel groups.
            worker_use_ray: Whether to use Ray for model workers. Will be set to
                True if either pipeline_parallel_size or tensor_parallel_size is
                greater than 1.
        """
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.worker_use_ray = worker_use_ray
        self.max_parallel_loading_workers = max_parallel_loading_workers

        self.world_size = pipeline_parallel_size * tensor_parallel_size * data_parallel_size

        self.num_workers = self.world_size + 1 # 1 worker for prepare stage
        self.num_cpus_extra_worker = num_cpus_extra_worker
        
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()


    def _verify_args(self) -> None:
        if (self.pipeline_parallel_size > 1 or self.tensor_parallel_size > 1 
            or self.data_parallel_size > 1):
            raise NotImplementedError(
                "Parallelism is not supported yet.")


class SchedulerConfig:
    """Scheduler config class."""   
    def __init__(
        self, 
        max_bathsize: int,
        use_mixed_precision: bool, 
        policy: str,
        overlap_prepare: bool,
    ) -> None:

        self.max_batchsize = max_bathsize
        self.use_mixed_precision = use_mixed_precision
        self.policy = policy
        self.overlap_prepare = overlap_prepare
        
        self._verify_args()
        
    def _verify_args(self) -> None:
        if self.max_batchsize <= 0:
            raise ValueError(f"Invalid max bathsize={self.max_batchsize}")


class EngineConfig:
    def __init__(
        self,
        log_status: bool,
        non_blocking_step: bool,
        engine_use_ray: bool = False,
        log_requests: bool = False,
    ) -> None:
        self.log_status = log_status
        self.non_blocking_step = non_blocking_step
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
    
    
    def verify_with_scheduler_config(self, scheduler_config: SchedulerConfig):
        # Currently we only support 2 combinations:
        # 1. blocking + non-overlapped
        # 2. nonblocking + overlapped
        assert self.non_blocking_step == scheduler_config.overlap_prepare