
from typing import Any, Optional, TYPE_CHECKING, Tuple

from sduss.utils import get_open_port
from sduss.logger import init_logger
from sduss.config import ParallelConfig, SchedulerConfig

logger = init_logger(__name__)

try:
    import ray
    from ray.air.util.torch_dist import TorchDistributedWorker
    # ! This API is not thoroughly studied
    class RayExecutor(TorchDistributedWorker):
        """Ray Wrapper for worker.
        
        Allowing worker to be lazily initialized after ray sets CUDA_VISIBLE_DEVICE.
        """
        
        def __init__(self, init_cached_hf_modules=False) -> None:
            if init_cached_hf_modules:
                # ? why here create cache directory and add it to python path?
                from transformers.dynamic_module_utils import init_hf_modules
                init_hf_modules()
            self.worker = None
        
        def init_worker(self, worker_init_fn) -> None:
            self.worker = worker_init_fn()
        
        def __getattr__(self, name: str) -> Any:
            return getattr(self.worker, name)
        
        def execute_method(self, method: str, *method_args, **method_kwargs):
            task_handler = getattr(self, method)
            return task_handler(*method_args, **method_kwargs)
except ImportError as e:
    logger.warning(f"Failed to import Ray with {e!r}. "
                   "For distributed inference, please install Ray with "
                   "`pip install ray pandas pyarrow`.")
    ray = None
    TorchDistributedWorker = None
    RayExecutor = None
    
if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup
    
def initialize_cluster(
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig,
    engine_use_ray: bool = False,
    ray_address: Optional[str] = None,
) -> Tuple[str, Optional["PlacementGroup"]]:
    """Initialize the distributed cluster with ray.
    
    We will check the number of available gpus in the cluster and the
    number specified by the parallel utils.

    Returns:
        A tuple of (`distributed_init_method`, `placement_group`). The
        `distributed_init_method` is the address for initializing the
        distributed backend. `placement_group` includes the specification
        of the resources for each distributed worker.
    """
    if parallel_config.worker_use_ray or engine_use_ray:
        if ray is None:
            raise ImportError("Ray is not properly installed! It is necessary "
                              "for distributed inference")
        
        ray.init(address=ray_address,
                 num_cpus=(parallel_config.world_size * parallel_config.num_cpus_gpu_worker
                            + (parallel_config.num_workers - parallel_config.world_size) * parallel_config.num_cpus_cpu_worker +
                            + 1 * 1),
                 num_gpus=parallel_config.world_size,
                 ignore_reinit_error=True)
    
    if not parallel_config.worker_use_ray:
        # Initialize cluster locally
        port = get_open_port()
        distributed_init_method = f"tcp://localhost:{port}"
        return distributed_init_method, None, None
    
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        bundles = current_placement_group.bundle_specs
        gpu_bundles = 0
        for bundle in bundles:
            # bundle is a dict
            bundle_gpus = bundle.get("GPU", 0)
            if bundle_gpus > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 GPU.")
            if bundle_gpus:
                gpu_bundles += 1
        if parallel_config.world_size > gpu_bundles:
            raise ValueError(
                "The number of required GPUs exceeds the total number of available GPUs "
                "in the cluster.")
        return (None, current_placement_group, None)
    else:
        num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
        if parallel_config.world_size > num_gpus_in_cluster:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the cluster.")

        # Create a new placement group
        # bundles = [{"GPU": 1, "CPU": parallel_config.num_cpus_gpu_worker}] * parallel_config.world_size
        bundles = [{"GPU": 1, "CPU": parallel_config.num_cpus_gpu_worker}] * parallel_config.world_size
        gpu_pg = ray.util.placement_group(bundles, strategy="STRICT_PACK")

        cpu_pg = None
        if scheduler_config.overlap_prepare:
            # We need extra workers
            # 1 more for the engine
            bundles = [{"CPU": 1}]
            bundles += [{"CPU": parallel_config.num_cpus_cpu_worker}] * (parallel_config.num_workers 
                                        - parallel_config.world_size)
            cpu_pg = ray.util.placement_group(bundles, strategy="STRICT_PACK")

        # We should wait until PG is ready -- this will block until all 
        # requested resources are available, and will timeout if 
        # they cannot be provisioned
        logger.debug("Start ray placement group allocation.")
        ray.get(gpu_pg.ready(), timeout=1800)
        if scheduler_config.overlap_prepare:
            ray.get(cpu_pg.ready(), timeout=1800)
        logger.debug("Ray plamencement group ready.")
        return (None, gpu_pg, cpu_pg)
    
