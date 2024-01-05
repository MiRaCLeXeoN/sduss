import time
from typing import Any, Dict, List, Union, Tuple

import torch
import numpy as np

from torch import nn

from sduss.config import ModelConfig, ParallelConfig, SchedulerConfig
from sduss.worker.cache_engine import KVCache
from sduss.model_executor import InputMetadata, get_model, SamplingMetadata
from sduss.sequence import SequenceGroupMetadata, SequenceData, SamplerOutput
from sduss.sampling_params import SamplingParams, SamplingType
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
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ):
        
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.model = None  # Set in load_model
        self.block_size = None  # Set after initial profiling.

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None else 0)
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
        self.model = get_model(self.model_config)
        
    def set_block_size(self, block_size: int) -> None:
        self. block_size = block_size
        max_num_blocks = (self.max_context_len_to_capture + block_size - 1) // block_size
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)
        
    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        """Prepare input data for prompt stage

        Args:
            seq_group_metadata_list (List[SequenceGroupMetadata]): Metadata of
                input sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, InputMetadata]: [input_token_ids,
                input_positions, input_metadata]
        """
        assert len(seq_group_metadata_list) > 0, "Empty input list!"
        # Data are filled in the same order of sequences ordered
        # in the seq_group_metadata_list
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mappings: List[List[int]] = []
        prompt_lens: List[int] = []
        
        for seq_group_metadata in seq_group_metadata_list:
            # Verify prompt stage
            assert seq_group_metadata.is_prompt, "Unexpected decoding sequence detected."
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1, "Unexpected multiple sequences in prompt stage."
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            
            prompt_lens.append(prompt_len)
            input_tokens.append(prompt_tokens)
            input_positions.append(list(range(prompt_len)))
            # During profiling, the tables are not initialized yet
            if seq_group_metadata.block_tables is None:
                slot_mappings.append([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping
            slot_mappings.append([])
            block_table = seq_group_metadata.block_tables[seq_id]
            # The block-reusing is implemented in block_manager.allocate()
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, prompt_len - self.sliding_window)
            for i in range(prompt_len):
                if i < start_idx:
                    slot_mappings[-1].append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mappings[-1].append(slot)

        # Prepare input data
        max_prompt_len = max(prompt_len)  # For padding
        input_tokens = _make_tensor_with_pad(input_tokens,
                                             max_prompt_len,
                                             pad=0,
                                             dtype=torch.long)
        input_positions = _make_tensor_with_pad(input_positions,
                                                max_prompt_len,
                                                pad=0,
                                                dtype=torch.long)
        slot_mappings = _make_tensor_with_pad(slot_mappings,
                                             max_prompt_len,
                                             pad=_PAD_SLOT_ID,
                                             dtype=torch.long)
        # ? Why at prompt stage, don't we use cuda graph?
        # ? Why no block_tables?
        input_metadata = InputMetadata(
            prompt_lens=prompt_lens,
            slot_mapping=slot_mappings,
            max_context_len=None,
            context_lens=None,
            block_tables=None,
            use_cuda_graph=False,
        )
        return input_tokens, input_positions, input_metadata
            
    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        """Prepare sequences for decoding stage.

        Args:
            seq_group_metadata_list (List[SequenceGroupMetadata]): Input seq_group_metadata

        Returns:
            Tuple[torch.Tensor, torch.Tensor, InputMetadata]: [input_token_ids,
                input_positions, input_metadata]
        """
        assert len(seq_group_metadata_list) > 0, "Empty Input list!"
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mappings: List[List[int]] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt, ("Unexpected prompt"
                "stage sequence detected")
            
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                # Input token
                seq_data = seq_group_metadata.seq_data[seq_id]
                last_token = seq_data.get_last_token_id()
                input_tokens.append([last_token])
                # Input position
                seq_len = seq_data.get_len()
                position = seq_len - 1  # Starts from 0
                input_positions.append([position])
                # Context length
                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)
                # Slot mapping
                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mappings.append([slot])
                # Block table
                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        batch_size = len(input_tokens)
        max_context_len = max(context_lens)
        use_captured_graph = (
            not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_context_len <= self.max_context_len_to_capture)
        
        if use_captured_graph:
            # Pad the input tokens, positions, and slot mapping to match the
            # batch size of the captured graph.
            graph_batch_size = _get_graph_batch_size(batch_size)
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append([])
                input_positions.append([])
                slot_mappings.append([])
                context_lens.append(1)
                block_tables.append([])
            batch_size = graph_batch_size

        # When using CUDA graph, we don't need to make the tensors on the GPU
        # because they will be copied to the designated GPU buffer in CUDAGraph.forward()
        device = "cpu" if use_captured_graph else "cuda"
        pin_memory = use_captured_graph and not self.in_wsl
        input_tokens = _make_tensor_with_pad(input_tokens,
                                             max_len=1,
                                             pad=0,
                                             dtype=torch.long,
                                             device=device,
                                             pin_memory=pin_memory)
        input_positions = _make_tensor_with_pad(input_positions,
                                                max_len=1,
                                                pad=0,
                                                dtype=torch.long,
                                                device=device,
                                                pin_memory=pin_memory)
        slot_mapping = _make_tensor_with_pad(slot_mapping,
                                             max_len=1,
                                             pad=_PAD_SLOT_ID,
                                             dtype=torch.long,
                                             device=device,
                                             pin_memory=pin_memory)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=device,
                                    pin_memory=pin_memory)
        
        if use_captured_graph:
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table  # in place
            # ? Why we have to use graph_block_tables???
            block_tables = torch.tensor(input_block_tables, device=device)
        else:
            # ? Why pad to max_context_len? it's 4 times longer!
            block_tables = _make_tensor_with_pad(
                block_tables,
                max_len=max_context_len,
                pad=0,
                dtype=torch.int,
            )

        input_metadata = InputMetadata(
            prompt_lens=[],
            slot_mapping=slot_mapping,
            max_context_len=max_context_len,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )
        return input_tokens, input_positions, input_metadata

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices: Dict[int, List[int]] = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0
        max_prompt_len = max(prompt_lens) if prompt_lens else 1
        
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1, "Unexpected decode stage sequence detected"
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += prompt_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        categorized_sample_indices_start_idx)
                categorized_sample_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += max_prompt_len
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        range(categorized_sample_indices_start_idx,
                              categorized_sample_indices_start_idx + num_seqs))
                categorized_sample_indices_start_idx += num_seqs

        selected_token_indices = _async_host2device(selected_token_indices,
                                            dtype=torch.long,
                                            pin_memory=not self.in_wsl)
        categorized_sample_indices = {
            t: _async_host2device(seq_ids, dtype=torch.int, pin_memory=not self.in_wsl)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
        )
        return sampling_metadata
                
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
            model_executable = self.model
        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
        )

        # Sample next token
        sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                 input_metadata.prompt_lens)
        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )
        return output
        
    @torch.inference_mode()
    def capture_model(self, kv_caches: List[KVCache]) -> None:
        """Capture the models using CUDAGraph with different batch sizes"""
        if self.model_config.enforce_eager:
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

            graph_runner = CUDAGraphRunner(self.model)
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
        self.graph: torch.cuda.CUDAGraph
        # mapping: name -> tensors as buffer
        self.input_buffers = Dict[str, torch.Tensor] = {}
        self.output_buffers = Dict[str, torch.Tensor] = {}
        
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
        self.graph = torch.nn.CUDAGraph()
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

def _get_graph_batch_size(self, batch_size: int) -> int:
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
    return torch.tensor(data, dtype, pin_memory
                        ).to(device="cuda", non_blocking=True)