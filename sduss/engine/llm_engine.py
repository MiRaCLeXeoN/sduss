"""Main engine module.

Here defines the main base Engine class

"""
import copy
import os
import time

from typing import Optional, Union, List, Any, Tuple, Dict, TYPE_CHECKING, Iterable
from functools import partial

import ray
# default to regard ray as an indispensible part
from ray.air.util.torch_dist import init_torch_dist_process_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sduss.logger import init_logger
from sduss.outputs import RequestOutputs
from sduss.sampling_params import SamplingParams
from sduss.utils import Counter
from sduss.config import (ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig)
from sduss.transformer_utils.tokenizer import get_tokenizer
from sduss.engine.ray_utils import RayWorker
from sduss.engine.metrics import record_metrics
from sduss.core.scheduler import Scheduler, SchedulerOutputs
from sduss.sequence import (SequenceStatus, 
                            Sequence, SequenceGroup, SequenceGroupMetadata,
                            SequenceOutputs, SequenceGroupOutputs, SamplerOutput)

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5

class LLMEngine:
    """The main engine that receives requests and generates texts.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_states: bool,
    ) -> None:
        """_summary_

        Args:
            model_config (ModelConfig): As name indicates
            cache_config (CacheConfig): As name indicates
            parallel_config (ParallelConfig): As name indicates
            scheduler_config (SchedulerConfig): As name indicates
            distributed_init_method (str): The initialization method for distributed
                execution. See `torch.distributed.init_process_group` for details.
            placement_group (Optional[PlacementGroup]): Ray placement group
                for distributed execution.
            log_status (bool): Whether to log statistics.
        """
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"seed={model_config.seed})") 
        
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_states = log_states
        
        self._verify_args()
        
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            tokenizer_revision=model_config.tokenizer_revision,
            revision=model_config.revision
        )
        # ? What's this counter for?
        self.seq_counter = Counter()
        
        # Create the parallel GPU workers
        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)
            
        # Profile the memory usage and initialize the cache
        # self._init_cache()
        
        self.scheduler = Scheduler(scheduler_config, cache_config)
        
        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []
        
    def _verify_args(self):
        """Verify args. Now only parallel config requires verification."""
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
    
    def _init_workers(self, distributed_init_method: str):
        """Initialize workers without ray
        
        Attach self.workers to self and call `init_model` method on all workers.

        Args:
            distributed_init_method (str): 
        """
        # ? Lazy import the worker to avoid importing torch.cude/xformers
        # ? before CUDA_VISIBLE_DEVICE is set in the worker
        from sduss.worker.worker import Worker

        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel size is greater than 1"
        )
        
        self.workers: List[Worker] = []
        worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        # initialize model on all workers
        self._run_workers("init_model", get_all_outputs=True)
        self._run_workers("load_model", get_all_outputs=True)
        
    def _init_workers_ray(
        self,
        placement_group: "PlacementGroup",
        **ray_remote_kwargs,
    ):        
        # Disable Ray usage stats collection
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # ? why should PlacementGroup use forward reference?
        
        # ? Lazy import the worker to avoid importing torch.cude/xformers
        # ? before CUDA_VISIBLE_DEVICE is set in the worker
        from sduss.worker.worker import Worker
        
        # create workers using ray interface
        # ! This ray API is not thoroughly examined
        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
                **ray_remote_kwargs,
            )(RayWorker).remote(self.model_config.trust_remote_code)
            self.workers.append(worker)
            
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        
        # execute `init_worker` method of workers
        self._run_workers(
            "init_worker",
            get_all_outputs=True,
            worker_init_fn=lambda: Worker(
                model_config,
                parallel_config,
                scheduler_config,
                None,
                None,
            )
        )
        self._run_workers("init_model", get_all_outputs=True)
        self._run_workers("load_model", get_all_outputs=True)
    
    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        raise NotImplementedError("vllm part not implemented yet")

    def add_request(
        self,
        request_id: int,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool

        Tokenization is performed here.

        Args:
            request_id (int): _description_
            prompt (Optional[str]): _description_
            samping_params (SamplingParams): _description_
            prompt_token_ids (Optional[List[int]], optional): _description_. Defaults to None.
            arrival_time (Optional[float], optional): _description_. Defaults to None.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)
    
    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)
    
    def _schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs,
                                 List[RequestOutputs]]:
        """Scheduling for this round running.

        Returns:
            Tuple[List[SequenceGroupMetadata], SchedulerOutputs, List[RequestOutputs]]: 
                (scheduled sequence groups' meta data list, scheduler output,
                request output wrapper of all ignored sequence groups). Since ignored
                sequence groups won't run any more, they will be returned as outputs.
        """        
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        return seq_group_metadata_list, scheduler_outputs, [
            RequestOutputs.from_seq_group(seq_group)
            for seq_group in scheduler_outputs.ignored_seq_groups
        ]
    
    def step(self) -> List[RequestOutputs]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored
        
        output: SamplerOutput = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        
        return self._process_model_outputs(output, scheduler_outputs)
    
    def _process_sequence_group_outputs(
        self,
        seq_group: SequenceGroup,
        outputs: SequenceGroupOutputs,
    ) -> None:
        """_summary_

        Args:
            seq_group (SequenceGroup): _description_
            outputs (SequenceGroupOutputs): _description_
        """
        # Extract prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs
            
        #
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict: Dict[int, List[SequenceOutputs]] = {
            parent_seq.seq_id : []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        
    
    def _process_model_outputs(
        self,
        output: SamplerOutput,
        scheduler_outputs: SchedulerOutputs,
    ) -> List[RequestOutputs]:
        """Process the model outputs from sampler and wrap them as
        `RequestOutputs`.

        Args:
            output (SamplerOutput): Output from the sampler
            scheduler_outputs (SchedulerOutputs): Output from the scheduler

        Returns:
            List[RequestOutputs]: Request outputs
            
        Finished sequence groups are freed here.
        """
        
        # Update the scheduled sequence groups with the model outputs
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs(seq_group, outputs)
            
        # Free the finished sequence groups
        self.scheduler.free_finished_seq_groups()
        
        # Wraps sampler outputs as request_outputs
        request_outputs: List[RequestOutputs] = []
        for seq_group in (scheduled_seq_groups + scheduler_outputs.ignored_seq_groups):
            request_output = RequestOutputs.from_seq_group(seq_group)
            request_outputs.append(request_output)
        
        if self.log_states:
            self._log_system_states(scheduler_outputs.prompt_run,
                                    scheduler_outputs.num_batched_tokens)
        return request_outputs        
    
    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """runs the model on all workers

        Args:
            method (str): the name of the method to be executed
            get_all_outputs (bool, optional): Get results from all workers. 
                Defaults to False.
        """
        all_outputs = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)
            
            output = executor(*args, **kwargs)
            all_outputs.append(output)
        
        # get ray obj ref
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
            
        if get_all_outputs:
            return all_outputs

        # if all workers returns the same result, just return one of them
        output = all_outputs[0]
        for o in all_outputs[1:]:
            assert o == output, "Detected variance between workers' result"
        return output
    
    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()
    
    def _log_system_states(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.monotonic()
        
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))
            
        should_log = now - self.last_logging_time >= _LOGGING_INTERVAL_SEC
        if not should_log:
            return
        
        # Discard the old states
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens
                                  if now - t < _LOGGING_INTERVAL_SEC]
        self.num_generation_tokens = [(t, n) for t, n in self.num_generation_tokens
                                      if now - t < _LOGGING_INTERVAL_SEC]
        
        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n
                                   for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (
            self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (
                self.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        record_metrics(
            avg_prompt_throughput=avg_prompt_throughput,
            avg_generation_throughput=avg_generation_throughput,
            scheduler_running=len(self.scheduler.running),
            scheduler_swapped=len(self.scheduler.swapped),
            scheduler_waiting=len(self.scheduler.waiting),
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
        ) 
        
        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now
        