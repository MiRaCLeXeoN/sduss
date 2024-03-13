"""All configuration classes

Including ModelConfig, 
"""

import pathlib
import torch

from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig

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
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
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
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[str] = None,
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.max_context_len_to_capture = max_context_len_to_capture
        
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
        supported_quantization = ["awq", "gptq", "squeezellm"]
        rocm_not_supported_quantization = ["awq"]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        hf_quant_config = getattr(self.hf_config, "quantization_config", None)
        if hf_quant_config is not None:
            hf_quant_method = str(hf_quant_config["quant_method"]).lower()
            if self.quantization is None:
                self.quantization = hf_quant_method
            elif self.quantization != hf_quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({hf_quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")
            if is_hip(
            ) and self.quantization in rocm_not_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not supported "
                    f"in ROCm.")
            logger.warning(f"{self.quantization} quantization is not fully "
                           "optimized yet. The speed can be slower than "
                           "non-quantized models.")

    def _verify_cuda_graph(self) -> None:
        if self.max_context_len_to_capture is None:
            self.max_context_len_to_capture = self.max_model_len
        self.max_context_len_to_capture = min(self.max_context_len_to_capture,
                                              self.max_model_len)
        if (self.quantization in ["gptq", "squeezellm"]
                and not self.enforce_eager):
            # Related issue: https://github.com/vllm-project/vllm/issues/2147
            logger.warning(f"{self.quantization} does not support CUDA graph "
                           "yet. Disabling CUDA graph.")
            self.enforce_eager = True
    
    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        """Verify the model config with parallel config
        
        Total number of attention heads must be divisible by tensor parallel size;
        total number of hidden layers must be divisible by pipeline parallel size;

        Args:
            parallel_config (ParallelConfig): parallel configuration

        Raises:
            ValueError
        """
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads}) "
                f"must be divisible by tensor parallel size ({tensor_parallel_size})."
            )
        
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                f"must be divisible by pipeline parallel size ({pipeline_parallel_size})."
            )
            
    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size
    
    def get_head_size(self) -> int:
        """Get the average hidden size for one attention head."""        
        # FIXME: This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads
    
    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU worker considering TMP."""
        # ! This one is directly copied.
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False))
        if not new_decoder_arch_falcon and getattr(self.hf_config,
                                                   "multi_query", False):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1
        # For Falcon:
        if getattr(self.hf_config, "n_head_kv", None) is not None:
            return (self.hf_config.n_head_kv //
                    parallel_config.tensor_parallel_size)
        if getattr(self.hf_config, "num_kv_heads", None) is not None:
            return (self.hf_config.num_kv_heads //
                    parallel_config.tensor_parallel_size)
        # For LLaMA-2:
        if getattr(self.hf_config, "num_key_value_heads", None) is not None:
            return (self.hf_config.num_key_value_heads //
                    parallel_config.tensor_parallel_size)
        total_num_attention_heads = self.hf_config.num_attention_heads
        return total_num_attention_heads // parallel_config.tensor_parallel_size
    
    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        """Get number of layers per GPU worker considering PMP."""        
        total_layers = self.hf_config.num_hidden_layers
        return total_layers // parallel_config.pipeline_parallel_size
    
    def get_vocab_size(self) -> int:
        return self.hf_config.vocab_size
    
    def get_sliding_window(self) -> Optional[int]:
        return getattr(self.hf_config, "sliding_window", None)
    
    def _load_format_is_dummy(self) -> bool:
        return self.load_format == "dummy"
    

class CacheConfig:
    """Configuration for KV cache
    """
    
    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
        sliding_window: Optional[int] = None,
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
        self.sliding_window = sliding_window
        self._verify_args()
        
        # Will be set after profiling
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None
        
    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")
    
    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        
        # ? Why these two values are same?
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node
        
        msg = (f"{cpu_memory_usage / _GB:.2f} GB out of the "
               f"{total_cpu_memory / _GB:.2f} GB total CPU memory is allocated "
               "for swap")
        # ? Why this value is fixed?
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Swap taks too much of the total memory. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning(msg)
        
            
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
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet.")

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
        max_num_batched_tokens: int, 
        max_num_seqs: int,
        max_model_len: int,
        max_paddings: int,
    ) -> None:

        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # If max_model_len is too short, use 2048 as the default value for
            # higher throughput.
            self.max_num_batched_tokens = max(max_model_len, 2048)
             
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_paddings = max_paddings
        self._verify_args()
        
    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")
        
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
    
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            f"{possible_keys}. Assuming the model's maximum length is "
            f"{default_max_len}.")
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        assert "factor" in rope_scaling
        scaling_factor = rope_scaling["factor"]
        if rope_scaling["type"] == "yarn":
            derived_max_model_len = rope_scaling[
                "original_max_position_embeddings"]
        derived_max_model_len *= scaling_factor
    
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