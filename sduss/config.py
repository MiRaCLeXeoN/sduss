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
        self.worker_use_ray = worker_use_ray
        self.max_parallel_loading_workers = max_parallel_loading_workers

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()


    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1 or self.tensor_parallel_size > 1:
            raise NotImplementedError(
                "Parallelism is not supported yet.")


class SchedulerConfig:
    """Init method

    Args:
        max_num_batched_tokens (int): Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs (int): Maximum number of sequences to be processed in a 
            single iteration.
        max_model_len (int): Maximum length of a sequence (including prompt
            and generated text).
        max_paddings (int): Maximum number of paddings to be added to a batch.
    """   
    def __init__(
        self, 
        max_bathsize: int,
        use_mixed_precision: bool, 
    ) -> None:

        self.max_batchsize = max_bathsize
        self.use_mixed_precision = use_mixed_precision
        
        self._verify_args()
        
    def _verify_args(self) -> None:
        if self.max_batchsize <= 0:
            raise ValueError(f"Invalid max bathsize={self.max_batchsize}")