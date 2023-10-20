
from typing import Any, Optional, TYPE_CHECKING

from sduss.logger import init_logger

logger = init_logger(__name__)

try:
    import ray
    from ray.air.util.torch_dist import TorchDistributedWorker
    # ! This API is not thoroughly studied
    
    class RayWorker(TorchDistributedWorker):
        """Ray Wrapper for worker.
        
        Allowing worker to be lazily initialized after ray sets CUDA_VISIBLE_DEVICE.
        """
        
        def __init__(self, init_cached_hf_modules=False) -> None:
            if init_cached_hf_modules:
                # ! why here create cache directory and add it to python path
                from transformers.dynamic_module_utils import init_hf_modules
                init_hf_modules()
            self.worker = None
        
        def init_worker(self, worker_init_fn) -> None:
            self.worker = worker_init_fn()
        
        def __getattribute__(self, __name: str) -> Any:
            return getattr(self.worker, __name)
        
        def execute_method(self, method: str, *method_args, **method_kwargs):
            task_handler = getattr(self, method)
            return task_handler(*method_args, **method_kwargs)
        
except ImportError as e:
    logger.warning(f"Failed to import Ray with {e!r}. "
                   "For distributed inference, please install Ray with "
                   "`pip install ray pandas pyarrow`.")