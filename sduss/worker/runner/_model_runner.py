import time
import os

import torch
import torch.distributed as dist

from typing import Any, Dict, List, Union, Tuple, Type, TYPE_CHECKING, Optional
from collections import defaultdict

from sduss.logger import init_logger
from sduss.model_executor import set_random_seed
from .wrappers import RunnerRequest, RunnerRequestDictType, RunnerOutput, InferenceStage

if TYPE_CHECKING:
    from sduss.model_executor.diffusers import BasePipeline
    from sduss.config import PipelineConfig, ParallelConfig, SchedulerConfig
    from sduss.model_executor import get_pipeline
    from ..wrappers import WorkerRequest

logger = init_logger(__name__)


class _ModelRunner:
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
        is_prepare_worker: bool,
        rank: int,
        device_num: int,
        distributed_init_method: str,
    ):
        
        self.pipeline_config = pipeline_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.is_prepare_worker = is_prepare_worker

        self.rank = rank
        self.device_num = device_num
        self.distributed_init_method = distributed_init_method

        self.req_mapping: 'Dict[int, RunnerRequest]' = {}

        # Set afterwards
        self.pipeline: 'BasePipeline' = None  # Set in load_model
        self.utils_cls: Dict[str, Type] = None  # Set after load_model

        # Log
        self.cycle_counter = 0
    

    def init_dis_env(self) -> None:
        """Initialize model on designated device."""
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573

        # Check up
        assert self.is_prepare_worker == False

        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        # ? This env var set by Ray causes exceptions with graph building
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        
        self.device = torch.device(f"cuda:{self.device_num}")
        torch.cuda.set_device(self.device)

        _init_distributed_environment(self.parallel_config, self.rank, 
                                      self.distributed_init_method)
        
        set_random_seed(self.pipeline_config.seed)

        logger.debug(f"Worker rank={self.rank} local_rank={self.local_rank} complete initialization")
    
    
    def init_prepare(self) -> None:
        assert self.is_prepare_worker
        # rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
        # local_rank = int(os.getenv("LOCAL_RANK"))
        # logger.debug(f"rank={self.rank}, local_rank={local_rank}")
        set_random_seed(self.pipeline_config.seed)


    def load_model(self) -> None:
        self.pipeline: 'BasePipeline' = get_pipeline(self.pipeline_config, self.is_prepare_worker)
        if self.scheduler_config.use_mixed_precision and not self.pipeline.SUPPORT_MIXED_PRECISION:
            raise ValueError("This pipeline doesn't support mixed precision input!")
        
        self.pipeline.__post_init__()
        self.utils_cls = self.pipeline.get_sampling_params_cls().utils_cls

        if self.is_prepare_worker:
            logger.debug(f"pipeline to cpu")
            self.pipeline.to("cpu")
        else:
            logger.debug(f"pipeline to cuda:{torch.cuda.current_device()}")
            self.pipeline.to(torch.cuda.current_device())

        return None


    def exec_prepare_stage(
        self,
        req_ids: List[int],
        req_sps: List,
    ) -> RunnerOutput:
        """Execute prepare stage inference.
        
        Runner reqs will be created and tracked at this stage.

        Args:
            scheduler_reqs (List[Request]): _description_
        """
        # 1. Create runner reqs to track reqs.
        runner_reqs_dict = defaultdict(list)
        for req_id, req_sp in zip(req_ids, req_sps):
            # Create
            rr = RunnerRequest(req_id, req_sp)
            # Track
            self.req_mapping[req_id] = rr

            res = rr.sampling_params.resolution
            runner_reqs_dict[res].append(rr)
            
        # 2. Execute
        start_time = time.time()
        self._exec_prepare_stage(runner_reqs_dict)
        end_time = time.time()

        # 3. Create return wrapper
        return RunnerOutput(
            runner_reqs=None,
            stage=InferenceStage.PREPARE,
            start_time=start_time,
            end_time=end_time,
        )
        
    
    def exec_denoising_stage(
        self,
        req_ids: List[int],
        is_sliced: bool,
        patch_size: int,
    ) -> RunnerOutput:
        """Execute denoising stage.

        Requests that finishes denoising stage don't need to be returned. Scheduelr
        can track requests' status according to its data duplicates.

        Args:
            req_ids (List[int]): IDs of requests to execute one iteration.
        """
        # 1. Collect requests and wrap as dict
        runner_reqs_dict = defaultdict(list)
        for req_id in req_ids:
            rr = self.req_mapping[req_id]
            res = rr.sampling_params.resolution
            runner_reqs_dict[res].append(rr)

        # 2. Execute
        start_time = time.time()
        self._exec_denoising_stage(runner_reqs_dict, is_sliced, patch_size)
        end_time = time.time()

        return RunnerOutput(
            runner_reqs=None,
            stage=InferenceStage.DENOISING,
            start_time=start_time,
            end_time=end_time,
        )
    
    
    def exec_post_stage(
        self,
        req_ids: List[int],
    ) -> RunnerOutput:
        # 1. Collect requests and wrap as dict
        runner_reqs_dict: "RunnerRequestDictType" = defaultdict(list)
        for req_id in req_ids:
            rr = self.req_mapping[req_id]
            res = rr.sampling_params.resolution
            runner_reqs_dict[res].append(rr)

        start_time = time.time()
        self._exec_post_stage(runner_reqs_dict)
        end_time = time.time()

        # Create output
        output = RunnerOutput(
            runner_reqs=runner_reqs_dict, 
            stage=InferenceStage.POST,
            start_time=start_time,
            end_time=end_time,
        )

        # Remove finished requests
        self._remove_requests_by_id(req_ids)

        return output


    def shutdown(self) -> None:
        dist.destroy_process_group()
        

    @torch.inference_mode()
    def _exec_prepare_stage(
        self,
        runner_reqs: "RunnerRequestDictType",
    ) -> None:
        prepare_input_cls = self.utils_cls['prepare_input']
        input_dict = prepare_input_cls.prepare_prepare_input(runner_reqs)
        self.pipeline.prepare_inference(**input_dict)
        
    
    @torch.inference_mode()
    def _exec_denoising_stage(
        self,
        runner_reqs: "RunnerRequestDictType",
        is_sliced: bool,
        patch_size: int,
    ) -> None:
        step_input_cls = self.utils_cls['step_input']
        input_dict = step_input_cls.prepare_step_input(runner_reqs, is_sliced=is_sliced, patch_size=patch_size)
        self.pipeline.denoising_step(**input_dict)
        # We don't need to find out finished ones, since scheduler can predict
        # request's status accoding to its num_inference_steps
    

    @torch.inference_mode()
    def _exec_post_stage(
        self,
        runner_reqs: "RunnerRequestDictType",
    ) -> None:
        post_input_cls = self.utils_cls['post_input']
        input_dict = post_input_cls.prepare_post_input(runner_reqs)
        self.pipeline.post_inference(**input_dict)
    
    
    def _remove_requests_by_id(self, req_ids: Union[int, List[int]]):
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        for req_id in req_ids:
            del self.req_mapping[req_id]
    
        
    @torch.inference_mode()
    def _profile_run(self) -> None:
        pass

        
    @torch.inference_mode()
    def capture_model(self, kv_caches) -> None:
        """Capture the models using CUDAGraph with different batch sizes"""
        # ! Not used
        raise NotImplementedError("CUDA graph is not supoorted yet")
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


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """"Initialize the distributed environment"""
    
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size})."
            )
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized"
        )
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method
        )
    
    # warmup
    # tensor = torch.ones(1) * rank
    # tensor = tensor.cuda()
    # print(f"[Rank {rank}, device {torch.cuda.current_device()}] Before allreduce: {tensor.item()}")
    # torch.distributed.all_reduce(tensor)
    # print(f"[Rank {rank}, device {torch.cuda.current_device()}] After allreduce: {tensor.item()}")
    # initialize_model_parallel(parallel_config.tensor_parallel_size,
    #                           parallel_config.pipeline_parallel_size)