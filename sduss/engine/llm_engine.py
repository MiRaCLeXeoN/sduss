"""Main engine module.

Here defines the main base Engine class

"""
import copy
import ray

from typing import Optional, Union, List, Any, TYPE_CHECKING
from functools import partial

# default to regard ray as an indispensible part
from ray.air.util.torch_dist import init_torch_dist_process_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sduss.logger import init_logger
from sduss.utils import Counter
from sduss.config import (ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig)
from sduss.transformer_utils.tokenizer import get_tokenizer
from sduss.engine.ray_utils import RayWorker

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

class LLMEngine:
    """The main engine that receives requests and generates texts.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_states: bool,
    ) -> None:
        """_summary_

        Args:
            model_config (ModelConfig): As name indicates
            cache_config (CacheConfig): As name indicates
            parallel_config (ParallelConfig): As name indicates
            scheduler_config (SchedulerConfig): As name indicates
            distributed_init_method (str): The initialization method for distributed
                execution. See `torch.distributed.init_process_group` for details.
            placement_group (Optional[PlacementGroup]): Ray placement group
                for distributed execution.
            log_status (bool): Whether to log statistics.
        """
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"seed={model_config.seed})") 
        
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_states = log_states
        
        assert self.cache_config.sliding_window == getattr(
            self.model_config.hf_config, "sliding_window", None
        )

        self._verify_args()
        
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            tokenizer_revision=model_config.tokenizer_revision,
            revision=model_config.revision
        )
        # ? What's this counter for?
        self.seq_counter = Counter()
        
        # Create the parallel GPU workers
        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)
            
        # Profile the memory usage and initialize the cache
        # self._init_cache()
        
        self.scheduler = 
        
        
    
    def _verify_args(self):
        """Verify args. Now only parallel config requires verification."""
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
    
    def _init_workers(self, distributed_init_method: str):
        """Initialize workers without ray
        
        Attach self.workers to self and call `init_model` method on all workers.

        Args:
            distributed_init_method (str): 
        """
        
        # ? Lazy import the worker to avoid importing torch.cude/xformers
        # ? before CUDA_VISIBLE_DEVICE is set in the worker
        from sduss.worker.worker import Worker

        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel size is greater than 1"
        )
        
        self.workers: List[Worker] = []
        worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        # initialize model on all workers
        self._run_workers("init_model", get_all_outputs=True)
        
    def _init_workers_ray(
        self,
        placement_group: "PlacementGroup",
        **ray_remote_kwargs,
    ):
        # ? why should PlacementGroup use forward reference?
        
        # ? Lazy import the worker to avoid importing torch.cude/xformers
        # ? before CUDA_VISIBLE_DEVICE is set in the worker
        from sduss.worker.worker import Worker
        
        # create workers using ray interface
        # ! This ray API is not thoroughly examined
        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
                **ray_remote_kwargs,
            )(RayWorker).remote(self.model_config.trust_remote_code)
            self.workers.append(worker)
            
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        
        # execute `init_worker` method of workers
        self._run_workers(
            self.workers,
            get_all_outputs=True,
            worker_init_fn=lambda: Worker(
                model_config,
                parallel_config,
                scheduler_config,
                None,
                None,
            )
        )
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )
    
    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        raise NotImplementedError("vllm part not implemented yet")
        
    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """runs the model on all workers

        Args:
            method (str): the name of the method to be executed
            get_all_outputs (bool, optional): Get results from all workers. 
                Defaults to False.
        """
        all_outputs = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)
            
            output = executor(*args, **kwargs)
            all_outputs.append(output)
        
        # get ray obj ref
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
            
        if get_all_outputs:
            return all_outputs

        # if all workers returns the same result, just return one of them
        output = all_outputs[0]
        for o in all_outputs[1:]:
            assert o == output, "Detected variance between workers' result"
        return output
        