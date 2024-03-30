import time
from typing import Any, Dict, List, Union, Tuple

import torch
import numpy as np

from torch import nn

from .wrappers import WorkerRequest

from sduss.config import PipelineConfig, ParallelConfig, SchedulerConfig
from sduss.model_executor import get_pipeline
from sduss.utils import in_wsl
from sduss.logger import init_logger

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

        self.pipeline = None  # Set in load_model

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables: np.ndarray = None  # Set after initial profiling.
        # cache in_wsl result
        self.in_wsl = in_wsl()


    def load_model(self) -> None:
        self.pipeline = get_pipeline(self.pipeline_config)
        self.pipeline.to("cuda")


    @torch.inference_mode()
    def exec_prepare_stage(
        self,
        worker_reqs: List[WorkerRequest]
    ) -> None:
        for wq in worker_reqs:
            prepare_output = self.pipeline.prepare_inference(wq.sampling_params)
            # Store prepare output
            wq.prepare_output = prepare_output
            # Produce step input for denoising
            wq.step_input = wq.sampling_params.utils_cls["step_input"].from_prepare_output(prepare_output)
    
    @torch
    
        
    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> SamplerOutput:
        # All sequences should be in either prompt or decode stage
        is_prompt = seq_group_metadata_list[0].is_prompt
        
        # Prepare input
        if is_prompt:
            inputs = self._prepare_prompt(seq_group_metadata_list)
        else:
            inputs = self._prepare_decode(seq_group_metadata_list)
        input_tokens, input_positions, input_metadata = inputs 

        # Execute model
        if input_metadata.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.pipeline
        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
        )

        # Sample next token
        sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                 input_metadata.prompt_lens)
        output = self.pipeline.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )
        return output


    @torch.inference_mode()
    def profile_run(self) -> None:
        # We need to enable top-k to reflect the accurate memeory usage
        vocab_size = self.pipeline_config.get_vocab_size()
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        
        # Prepare input dummy sequences
        seqs: List[SequenceGroupMetadata] = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)
            
        # Run
        num_layers = self.pipeline_config.get_num_layers(self.parallel_config)
        kv_caches = [(None, None)] * num_layers
        self.execute_model(seqs, kv_caches=kv_caches)
        torch.cuda.synchronize()

        
    @torch.inference_mode()
    def capture_model(self, kv_caches: List[KVCache]) -> None:
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
            input_metadata = InputMetadata(
                prompt_lens=[],
                slot_mapping=slot_mapping[:batch_size],
                max_context_len=self.max_context_len_to_capture,
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
                use_cuda_graph=True,
            )

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
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        memory_pool,
    ) -> None:
        assert self.graph is None
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        self.model(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
        )
        torch.cuda.synchronize()
        
        # Capture the graph
        self.graph = torch.cuda.CUDAGraph()
        # Pass the previous memory pool to share it among the graphs
        with torch.cuda.graph(self.graph, pool=memory_pool):
            hidden_states = self.model(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
            )
        torch.cuda.synchronize()
        
        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": input_metadata.slot_mapping,
            "context_lens": input_metadata.context_lens,
            "block_tables": input_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Fill the buffers with new data
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(input_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["context_lens"].copy_(input_metadata.context_lens,
                                                 non_blocking=True)
        self.input_buffers["block_tables"].copy_(input_metadata.block_tables,
                                                 non_blocking=True)

        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

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