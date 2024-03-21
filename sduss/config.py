"""All configuration classes """
from typing import Optional, Union

from sduss.logger import init_logger
from sduss.utils import get_cpu_memory
from sduss.transformer_utils.config import get_config
from sduss.utils import is_hip


logger = init_logger(__name__)

_GB = 1 << 30
class ModelConfig:
    """Configuration for model.
    
    Models are default to use from huggingface models.

    Args:
        model (str): Name or path of the huggingface model to use.
        trust_remote_code (bool): Trust code from remote, e.g. from Huggingface, when
            downloading the model and tokenizer
        download_dir (str): Path to download and load the weights, default to the default
            cache directory of huggingface
        load_format (str): The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format. More about
                safetensors to be seen at https://github.com/huggingface/safetensors
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
        dtype (str): data type for model weights and activations
        seed (int): _description_
        revision (Optional[str], optional): The specific model version to use.
            Defaults to None.
    """
    
    def __init__(
        self,
        model: str,
        trust_remote_code: bool,
        download_dir: str,
        load_format: str,
        dtype: str,
        seed: int,
        revision: Optional[str] = None,
    ) -> None:
        self.model = model
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.dtype = dtype
        self.seed = seed
        self.revision = revision
        
        self._verify_load_format()
        
    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        if load_format not in [
                "auto", "pt", "safetensors", "npcache", "dummy"
        ]:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'.")
        self.load_format = load_format
        
    def _load_format_is_dummy(self) -> bool:
        return self.load_format == "dummy"
    

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
    ) -> None:

        self.max_batchsize = max_bathsize
        
        self._verify_args()
        
    def _verify_args(self) -> None:
        if self.max_batchsize <= 0:
            raise ValueError(f"Invalid max bathsize={self.max_batchsize}")