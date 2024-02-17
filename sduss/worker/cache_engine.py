from typing import Tuple, List, Dict

import torch

# ! cache_ops is not currently available via sduss
from vllm import cache_ops
from sduss.config import CacheConfig, ModelConfig, ParallelConfig
from sduss.utils import in_wsl, get_dtype_size
from sduss.logger import init_logger

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]

class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    
    CacheEngine must be initialized after profiling. Otherwise essential data
    will be absent.
    """
    
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        
        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)
        self.dtype =model_config.dtype
        
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()
        
        # Initialize a cuda stream for cache operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        
        # Cuda events for stream synchronizations.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]
        
    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        # ? Why use 16 bytes as the minimal unit?
        element_size = get_dtype_size(self.dtype)
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x
        )
    
    def get_value_block_shape(self) -> Tuple[int, int, int]:
        # ? Why different from key?
        return (
            self.num_heads,
            self.head_size,
            self.block_size
        )
    
    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        # ? Why depends on number of layers?
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache
        
    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        """Swap KV cache blocks from src to dst.

        Args:
            src (List[KVCache]): Source
            dst (List[KVCache]): Destination
            src_to_dst (Dict[int, int]): Mapping from src to dst.
        """
        # ? Why we have to specify a cuda stream here?
        # ? Why do we need events?
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)
                
                event = self.events[i]
                event.record(stream=self.cache_stream) 
                
    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)
    
    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)
               
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
    
    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        """Get the size of all cache blocks together of current worker in bytes.

        Args:
            block_size (int): _description_
            model_config (ModelConfig): _description_
            parallel_config (ParallelConfig): _description_

        Returns:
            int: _description_
        """        
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = get_dtype_size(model_config.dtype)
        return dtype_size * total
        