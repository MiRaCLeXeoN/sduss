import time
from typing import Any, Dict, List, Union, Tuple, Type

import torch
import numpy as np

from torch import nn

from .wrappers import WorkerRequest

from sduss.config import PipelineConfig, ParallelConfig, SchedulerConfig
from sduss.model_executor import get_pipeline
from sduss.utils import in_wsl
from sduss.logger import init_logger
from sduss.model_executor.diffusers import BasePipeline

logger = init_logger(__name__)

_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]
_PAD_SLOT_ID = -1

class ModelRunner:
    """The model runner is responsible for all model-relevant
    operations in a worker.

    Args:
        model_config (ModelConfig): 
        parallel_config (ParallelConfig): 
        scheduler_config (SchedulerConfig): 
    """
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ):
        
        self.pipeline_config = pipeline_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config

        self.pipeline: BasePipeline = None  # Set in load_model

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        # Set after load_model
        self.utils_cls: Dict[str, Type] = None


    def load_model(self) -> None:
        self.pipeline: BasePipeline = get_pipeline(self.pipeline_config)
        if self.scheduler_config.use_mixed_precision and not self.pipeline.SUPPORT_MIXED_PRECISION:
            raise ValueError("This pipeline doesn't support mixed precision input!")

        self.pipeline.to("cuda")
        self.utils_cls = self.pipeline.get_sampling_params_cls().utils_cls
        

    @torch.inference_mode()
    def exec_prepare_stage(
        self,
        worker_reqs: List[WorkerRequest],
    ) -> None:
        prepare_input_cls = self.utils_cls['prepare_input']
        input_dict = prepare_input_cls.prepare_prepare_input(worker_reqs)
        self.pipeline.prepare_inference(**input_dict)
        
    
    @torch.inference_mode()
    def exec_denoising_stage(
        self,
        worker_reqs: List[WorkerRequest],
    ) -> None:
        step_input_cls = self.utils_cls['step_input']
        input_dict = step_input_cls.prepare_step_input(worker_reqs)
        self.pipeline.denoising_step(**input_dict)
        # We don't need to find out finished ones, since scheduler can predict
        # request's status accoding to its num_inference_steps
    

    @torch.inference_mode()
    def exec_post_stage(
        self,
        worker_reqs: List[WorkerRequest],
    ) -> None:
        post_input_cls = self.utils_cls['post_input']
        input_dict = post_input_cls.prepare_post_input(worker_reqs)
        self.pipeline.post_inference(**input_dict)
    
        
    @torch.inference_mode()
    def profile_run(self) -> None:
        pass

        
    @torch.inference_mode()
    def capture_model(self, kv_caches) -> None:
        """Capture the models using CUDAGraph with different batch sizes"""
        if self.pipeline_config.enforce_eager:
            raise RuntimeError("Trying to using cuda graph while "
                               f"setting enforce_eager")
        # ? How much additional memory does CUDA graph use?
        start_time = time.perf_counter()
        
        # Prepar dummy inputs. These will be reused for all batch sizes.
        # Use the max batchsize to ensure downward compatibility of memory pool
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, 1, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, 1, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        # ? Capturing the largest batch size first may help reduce the
        # ? memory usage of CUDA graph.
        for batch_size in reversed(_BATCH_SIZES_TO_CAPTURE):
            # Create dummy input_metadata.
            input_metadata = None

            graph_runner = CUDAGraphRunner(self.pipeline)
            graph_runner.capture(
                input_tokens[:batch_size],
                input_positions[:batch_size],
                kv_caches,
                input_metadata,
                memory_pool=self.graph_memory_pool,
            )
            # Save the current memory pool to pass to the next graph
            self.graph_memory_pool = graph_runner.graph.pool()
            self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")
        
        
        
class CUDAGraphRunner:
    """CUDA Graph wrapper

    Args:
        model (nn.Module): Bound model
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.graph: torch.cuda.CUDAGraph = None
        # mapping: name -> tensors as buffer
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}
        
    def capture(
        self,
    ) -> None:
        pass

    def forward(
        self,
    ) -> torch.Tensor:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    

def _pad_to_max(
    x: List[int],
    max_len: int,
    pad: int,
) -> List[int]:
    """Pad a List to `max_len` using `pad`

    Args:
        x (List[int]): Target list to pad
        max_len (int): Target length
        pad (int): Number used for padding

    Raises:
        ValueError: Trying to pad a list longer than `max_len`

    Returns:
        List[int]: Padded list
    """
    if len(x) > max_len:
        raise ValueError("Trying to pad an ineligible list!")
    return x + [pad] * (max_len - len(x))

def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cuda",
    pin_memory: bool = False,
) -> torch.Tensor:
    """Pad `x` along the most inner dimension to `max_len` using `pad`

    Args:
        x (List[List[int]]): Target List[List]
        max_len (int): Max length to pad to
        pad (int): Numder used for padding
        dtype (torch.dtype): Data type of returned Tensor
        device (Union[str, torch.device], optional): Defaults to "cuda".
        pin_memory (bool, optional): Whether to pin the tensor in memory,
            only applicable for cpu Tensors. Defaults to False.

    Returns:
        torch.Tensor: _description_
    """
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x, dtype=dtype, device=device,
                        pin_memory=pin_memory and str(device) == "cpu")

def _get_graph_batch_size(batch_size: int) -> int:
    """Get the appropriate batch size for inference

    Args:
        batch_size (int): The reference batch_size

    Returns:
        int: Only 1, 2, 4, n * 8 are legitimate
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return (batch_size + 7) // 8 * 8
    
def _async_host2device(
    data: List,
    dtype: torch.dtype,
    pin_memory: bool,
) -> torch.Tensor:
    """Asynchronously move the `data` from cpu to device

    Args:
        data (List): Target
        dtype (torch.dtype): Data type
        pin_memory (bool): Whether to pin in the memory

    Returns:
        torch.Tensor: The tensor on device
    """
    return torch.tensor(data, dtype=dtype, pin_memory=pin_memory
                        ).to(device="cuda", non_blocking=True)