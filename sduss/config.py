"""All configuration classes

Including ModelConfig, 
"""

import pathlib
import torch

from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig

from sduss.logger import init_logger
from sduss.transformer_utils.config import get_config

logger = init_logger(__name__)

_GB = 1 << 30
class ModelConfig:
    """Configuration for model.
    
    Models are default to use from huggingface models.
    
    Attributes:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: 
    
    """
    
    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        download_dir: str,
        load_format: str,
        dtype: str,
        seed: int,
        revision: Optional[str] = None,
        max_model_len: Optional[str] = None,
        quantization: Optional[str] = None,
    ) -> None:
        """init method

        Args:
            model (str): Name or path of the huggingface model to use.
            tokenizer (str): Name or path of the huggingface model to use.
            tokenizer_mode (str): Should be either 'auto' or 'slow'. 'auto' will use
                the fast tokenizer if available, and 'slow' will always use the slow one.
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
            max_sequence_len (Optional[str], optional): Maximum length of a sequence.
                Defaults to None.
            quantization_method (Optional[str], optional): Quantization method. Only
                'awq' and None are supported. Defaults to None.
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        
        self.hf_config = get_config(model, trust_remote_code, revision)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.max_model_len = _get_and_verify_max_len(self.hf_config, max_model_len)
        self._verify_load_format()
        self._verify_tokenizer_mode()
        self._verify_quantization()
        
    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        if load_format not in [
                "auto", "pt", "safetensors", "npcache", "dummy"
        ]:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'.")
        self.load_format = load_format
        
    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto' or 'slow'.")
        self.tokenizer_mode = tokenizer_mode

    def _verify_quantization(self) -> None:
        supported_quantization = ["awq"]
        if self.quantization is None:
            return
        quantization = self.quantization.lower()
        if quantization not in supported_quantization:
            raise ValueError(
                f"Unknown quantization: {self.quantization}. Must be one of "
                f"{supported_quantization}.")
        self.quantization = quantization

class CacheConfig:
    """Configuration for KV cache
    """
    
    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
    ) -> None:
        """init method

        Args:
            block_size (int): Size of a cache block in number of tokens.
            gpu_memory_utilization (float): Fraction of GPU memory to
                use for execution
            swap_space (int): Size of the CPU swap space per GPU in GB.
        """
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GB
        self._verify_args()
        
        # Will be set after profiling
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None
        
    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")
            
class ParallelConfig:
    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: bool,
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

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet.")

class SchedulerConfig:
    def __init__(self, max_num_batched_tokens: int, max_num_seqs: int,
                 max_model_len: int) -> None:
        """Init method

        Args:
            max_num_batched_tokens (int): Maximum number of tokens to be processed in
                a single iteration.
            max_num_seqs (int): Maximum number of sequences to be processed in a 
                single iteration.
            max_model_len (int): Maximum length of a sequence (including prompt
                and generated text).
        """        
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        
_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: str,
) -> torch.dtype:
    """verify data type according to the configuration

    Args:
        config (PretrainedConfig): PretrainedConfig
        dtype (str): a string representing data type, coule be either 'auto'
            or one in the _STR_DTYPE_TO_TORCH_DTYPE

    Returns:
        torch.dtype: data type verified
    """
    # the dtype used by the model
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32
    
    # convert the dtype to torch data type
    # torch_dtype denotes the final dtype to be used
    dtype = dtype.lower()
    if dtype == 'auto':
        if config_dtype == torch.float32:
            # float16 for float32 models
            torch_dtype = torch.float16
        else:
            torch_dtype = config_dtype
    else:
        if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {dtype}")
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    
    # verify the casting
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 is allowed
            pass
        else:
            # Other conditions should be warned
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")
    
    # check for GPU support
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                f"bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU compute capability is "
                f"{compute_capability[0]}.{compute_capability[1]}."
            )   
    return torch_dtype

def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
) -> int:
    """Get and verify the model's maximum length from PretrainedConfig"""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    for key in possible_keys:
        max_len_key = getattr(hf_config, key, None)
        if max_len_key is not None:
            derived_max_model_len = min(derived_max_model_len, max_len_key)
    
    if max_model_len is None:
        max_model_len = derived_max_model_len
    elif max_model_len > derived_max_model_len:
        raise ValueError(
            f"User-specified max_model_len ({max_model_len}) is greater than "
            f"the derived max_model_len ({max_len_key}={derived_max_model_len}"
            " in model's config.json). This may lead to incorrect model "
            "outputs or CUDA errors. Make sure the value is correct and "
            "within the model context size."
        )
    return max_model_len