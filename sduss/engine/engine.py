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

from sduss.scheduler import Scheduler, SchedulerOutput, Request, RequestStatus
from sduss.worker import WorkerOutput

from sduss.logger import init_logger
from sduss.outputs import RequestOutput
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.utils import Counter
from sduss.config import (PipelineConfig, ParallelConfig, SchedulerConfig)
from sduss.engine.arg_utils import EngineArgs
from sduss.engine.ray_utils import RayWorker, initialize_cluster
from sduss.engine.metrics import record_metrics


if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5

class Engine:
    """The main engine that receives requests and generates texts.
    """
    
    def __init__(
        self,
        pipeline_config: PipelineConfig,
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
            f"model={pipeline_config.pipeline!r}, "
            f"seed={pipeline_config.seed})") 
        
        self.pipeline_config = pipeline_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_states = log_states
        
        self._verify_args()
        
        # Create the parallel GPU workers
        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)
        
        self.scheduler = Scheduler(scheduler_config)
            
        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_generated_images: List[Tuple[float, int]] = []
    
    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "Engine":
        """Create an inference engine from arguments"""
        # Create engine configs.
        model_config, parallel_config, scheduler_config = engine_args.create_engine_configs()
        # Initialize the cluster
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config)
        # Create engine instance
        return cls(model_config, parallel_config, scheduler_config,
                   distributed_init_method, 
                   placement_group,
                   log_states=not engine_args.disable_log_status)
        

    def _verify_args(self):
        """Verify args. Now only parallel config requires verification."""
        self.pipeline_config.verify_with_scheduler_config(self.scheduler_config)

    
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
            self.pipeline_config,
            self.parallel_config,
            self.scheduler_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        # initialize model on all workers
        self._run_workers("init_dis_env", get_all_outputs=True)
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
        
        # ? Lazy import the worker to avoid importing torch.cuda/xformers
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
            )(RayWorker).remote(self.pipeline_config.trust_remote_code)
            self.workers.append(worker)
            
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.pipeline_config)
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
        self._run_workers("init_dis_env", get_all_outputs=True)
        self._run_workers("load_model", get_all_outputs=True)
    

    def add_request(
        self,
        request_id: int,
        sampling_params: BaseSamplingParams,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        Args:
            request_id (int): _description_
            prompt (Optional[str]): _description_
            samping_params (SamplingParams): _description_
            prompt_token_ids (Optional[List[int]], optional): _description_. Defaults to None.
            arrival_time (Optional[float], optional): _description_. Defaults to None.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()

        # Create a new Request
        req = Request(request_id=request_id, 
                      arrival_time=arrival_time, 
                      sampling_params=sampling_params)

        # Add the request to the scheduler.
        self.scheduler.add_request(req)
    
    def abort_request(self, request_id: Union[int, Iterable[int]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        self.scheduler.abort_request(request_id)

    
    def _schedule(self) -> Tuple[SchedulerOutput, List[int]] :
        """Scheduling for this round running."""        
        scheduler_outputs = self.scheduler.schedule()
        # Extract request ids
        req_ids = []
        for req in scheduler_outputs.scheduled_requests:
            req_ids.append(req.request_id)
        
        return scheduler_outputs, req_ids

    
    def step(self) -> List[RequestOutput]:
        """Performs one denoising iteration and returns newly generated results."""
        scheduler_output, req_ids = self._schedule()
        if scheduler_output.status == RequestStatus.WAITING:
            # For prepare stage inference
            self._run_workers("exec_prepare_stage", scheduler_output=scheduler_output)
        elif (scheduler_output.status == RequestStatus.PREPARE or
            scheduler_output.status == RequestStatus.DENOISING):
            # For denoising stage inference
            self._run_workers("exec_denoising_stage", req_ids=req_ids)
        elif (scheduler_output.status == RequestStatus.POSTPROCESSING):
            # For post stage inference
            output: WorkerOutput = self._run_workers(
                "execute_post_stage",
                scheduler_output=scheduler_output,
            )
        
        return self._process_output(scheduler_output, req_ids, output)

    
    def _process_output(
        self,
        scheduler_outputs: SchedulerOutput,
        req_ids: List[int],
        output: Optional[WorkerOutput],
    ) -> List[RequestOutput]:
        """Update requests status and prepare return result if available."""
        
        # Update the scheduled sequence groups with the model outputs
        scheduled_reqs = scheduler_outputs.scheduled_requests
        if scheduler_outputs.status == 
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs(seq_group, outputs)
            
        # Free the finished sequence groups
        self.scheduler.free_finished_requests()
        
        # Wraps sampler outputs as request_outputs
        request_outputs: List[RequestOutput] = []
        for seq_group in (scheduled_seq_groups + scheduler_outputs.ignored_seq_groups):
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        
        if self.log_states:
            self._log_system_states(scheduler_outputs.prompt_run,
                                    scheduler_outputs.num_batched_tokens)
        return request_outputs
    
    def _run_workers_in_batch(
        self,
        workers: List,
        method: str,
        *args,
        **kwargs,
    ):
        all_outputs = []
        for worker in workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)
        
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
        return all_outputs
    
    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the method on all workers

        Args:
            method (str): the name of the method to be executed
            get_all_outputs (bool, optional): Get results from all workers. 
                Defaults to False.
        """
        all_outputs = []
        if max_concurrent_workers:
            work_groups = [
                self.workers[i:i + max_concurrent_workers]
                for i in range(0, len(self.workers), max_concurrent_workers)
            ]
        else:
            work_groups = [self.workers]

        for workers in work_groups:
            all_outputs.extend(
                self._run_workers_in_batch(workers, method, *args, **kwargs)
            )
        
        if get_all_outputs:
            return all_outputs
        else:
            output = all_outputs[0]
            for other_output in all_outputs[1:]:
                assert output == other_output, "Trying to ignore other valid outputs."
            return output
    
    def get_model_config(self) -> PipelineConfig:
        """Gets the model configuration."""
        return self.pipeline_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_requests()
    
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
        