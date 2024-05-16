"""Main engine module.

Here defines the main base Engine class

"""
import copy
import os
import time
import datetime
import sys
import logging

from typing import Optional, Union, List, Any, Tuple, Dict, TYPE_CHECKING, Iterable
from functools import partial

import ray
# default to regard ray as an indispensible part
from ray.air.util.torch_dist import init_torch_dist_process_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sduss.scheduler import Scheduler, SchedulerOutput, Request, RequestStatus
from sduss.worker import WorkerOutput
from sduss.logger import init_logger
from sduss.entrypoints.outputs import RequestOutput
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.config import (PipelineConfig, ParallelConfig, SchedulerConfig, EngineConfig)
from sduss.engine.arg_utils import EngineArgs
from sduss.worker.ray_utils import RayWorker, initialize_cluster
from sduss.engine.metrics import record_metrics

from .utils import SmUtilMonitor

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup
    from sduss.model_executor.diffusers import BasePipeline

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
        engine_config: EngineConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
    ) -> None:
        if engine_config.log_status:
            logger.info(
                "Initializing an engine with config:\n"
                f"model={pipeline_config.pipeline!r}\n"
                f"seed={pipeline_config.seed}") 
        
        self.pipeline_config = pipeline_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.engine_config = engine_config

        self._verify_args()
        
        # Create the parallel GPU workers
        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)
            
        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_generated_images: List[Tuple[float, int]] = []

        # Non-blokcing required attributes
        self.prev_prepare_handlers = None
        self.prev_denoising_handlers = None
        self.prev_postprocessing_handlers = None
        self.prev_scheduler_output = None

        self.pipeline_cls = None
        self.sampling_param_cls = None
        self._set_pipeline_cls()
        
        self.support_resolutions = self.pipeline_cls.SUPPORT_RESOLUTIONS
        self.scheduler = Scheduler(scheduler_config, engine_config, self.support_resolutions)

        self._step_counter = 0
        self.engine_ready = True
        # Flust outputs so that we can see logs ASAP.
        if engine_config.log_status:
            logger.info("Engine initialization done. System ready.")
        
        # FIXME: For experiment
        self.collect_data = os.getenv("SDUSS_COLLECT_DATA")
        if self.collect_data:
            self._prepare_collect_data()

    
    def engine_is_ready(self) -> bool:
        return self.engine_ready
    
    
    def _set_pipeline_cls(self) -> None:
        from sduss.model_executor.model_loader import get_pipeline_cls
        self.pipeline_cls: BasePipeline = get_pipeline_cls(self.pipeline_config)
        self.sampling_param_cls = self.pipeline_cls.get_sampling_params_cls()

    
    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "Engine":
        """Create an inference engine from arguments"""
        # Create engine configs.
        pipeline_config, parallel_config, scheduler_config, engine_config= engine_args.create_engine_configs()
        # Initialize the cluster
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config, scheduler_config)
        # Create engine instance
        return cls(pipeline_config, 
                   parallel_config, 
                   scheduler_config, 
                   engine_config,
                   distributed_init_method, 
                   placement_group)
        

    def _verify_args(self):
        """Verify args. Now only parallel config requires verification."""
        self.pipeline_config.verify_with_scheduler_config(self.scheduler_config)
        self.engine_config.verify_with_scheduler_config(self.scheduler_config)

    
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
            self.engine_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        # initialize model on all workers
        self._run_workers_blocking("init_dis_env", get_all_outputs=True)
        self._run_workers_blocking("load_model", get_all_outputs=True)

        
    def _init_workers_ray(
        self,
        placement_group: "PlacementGroup",
        **ray_remote_kwargs,
    ):
        if self.engine_config.log_status:
            logger.info("_init_workers_ray called")
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
        self.workers = []
        self.prepare_workers = []
        for bundle in placement_group.bundle_specs:
            # if not bundle.get("GPU", 0):
            #     continue
            logger.info(f"bundle gpus={bundle.get('GPU')}, cpus={bundle.get('CPU')}")
            if bundle.get("GPU"):
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_capture_child_tasks=True),
                    **ray_remote_kwargs,
                )(RayWorker).remote(self.pipeline_config.trust_remote_code)
                self.workers.append(worker)
            elif bundle.get("CPU") == self.parallel_config.num_cpus_extra_worker:
                worker = ray.remote(
                    num_cpus=1,
                    num_gpus=0,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_capture_child_tasks=True),
                    **ray_remote_kwargs,
                )(RayWorker).remote(self.pipeline_config.trust_remote_code)
                self.prepare_workers.append(worker)
        
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.pipeline_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        engine_config = copy.deepcopy(self.engine_config)
        
        # execute `init_worker` method of workers
        self._run_workers_blocking(
            "init_worker",
            self.workers,
            get_all_outputs=True,
            worker_init_fn=lambda: Worker(
                model_config,
                parallel_config,
                scheduler_config,
                engine_config
            )
        )
        self._run_workers_blocking("init_dis_env", self.workers, get_all_outputs=True)
        # execute `init_worker` method of prepare_workers
        if self.scheduler_config.overlap_prepare:
            self._run_workers_blocking(
                "init_worker",
                self.prepare_workers,
                get_all_outputs=False,
                worker_init_fn=lambda: Worker(
                    model_config,
                    parallel_config,
                    scheduler_config,
                    engine_config,
                    is_prepare_worker=True,
                )
            )
            self._run_workers_blocking("init_prepare", self.prepare_workers, get_all_outputs=True)
            self._run_workers_blocking("load_model", self.workers + self.prepare_workers, get_all_outputs=True)
        else:
            self._run_workers_blocking("load_model", self.workers, get_all_outputs=True)
    

    def add_request(
        self,
        request_id: int,
        sampling_params: BaseSamplingParams,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        Args:
            request_id (int): _description_
            samping_params (SamplingParams): _description_
            arrival_time (Optional[float], optional): _description_. Defaults to None.
        """
        # Create a new Request
        req = Request(request_id=request_id, 
                      arrival_time=arrival_time, 
                      sampling_params=sampling_params)

        # Add the request to the scheduler.
        self.scheduler.add_request(req)
    
    
    def add_request_batch(
        self,
        new_requests_params: List[Dict],
    ) -> None:
        """Add a batch of requests."""
        for req_param_dict in new_requests_params:
            req = Request(**req_param_dict)
            self.scheduler.add_request(req)

    
    def abort_requests(self, request_ids: Union[int, Iterable[int]]) -> List[Request]:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        return self.scheduler.abort_requests(request_ids)

    
    def _schedule(self) -> Tuple[SchedulerOutput, List[int]] :
        """Scheduling for current round."""
        if self.scheduler_config.overlap_prepare:
            scheduler_output = self.scheduler.schedule_overlap_prepare()
        else:
            scheduler_output = self.scheduler.schedule()
        # Extract request ids
        req_ids = scheduler_output.get_req_ids()
        
        return scheduler_output, req_ids
    
    
    def _step_blocking(self) -> List[RequestOutput]:
        """Performs one denoising iteration and returns newly generated results."""
        scheduler_output, req_ids = self._schedule()

        if self.engine_config.log_status:
            self._log_system_states(scheduler_output)

        output = None
        if scheduler_output.status == RequestStatus.WAITING:
            # Currently, we don't do anything in waiting stage
            pass
        elif scheduler_output.status == RequestStatus.PREPARE:
            # For prepare stage inference
            # TODO(MX): We may pass schduler output directly
            output: WorkerOutput = self._run_workers_blocking(
                "exec_prepare_stage", 
                self.workers,
                scheduler_reqs=scheduler_output.get_reqs_as_list(),
                use_mixed_precision=self.scheduler_config.use_mixed_precision)
        elif scheduler_output.status == RequestStatus.DENOISING:
            # For denoising stage inference
            output: WorkerOutput = self._run_workers_blocking(
                "exec_denoising_stage", 
                self.workers,
                req_ids=req_ids,
                use_mixed_precision=self.scheduler_config.use_mixed_precision,
                is_sliced=scheduler_output.is_sliced,
                patch_size=scheduler_output.patch_size)
        elif (scheduler_output.status == RequestStatus.POSTPROCESSING):
            # For post stage inference
            output: WorkerOutput = self._run_workers_blocking(
                "exec_post_stage",
                self.workers,
                req_ids=req_ids,
                use_mixed_precision=self.scheduler_config.use_mixed_precision,
            )
        else:
            # We don't expect EMPTY to be in blocking method
            raise RuntimeError(f"Unexpected status {scheduler_output.status}.")
        
        finished_req_outputs = self._process_output(scheduler_output=scheduler_output,
                                       req_ids=req_ids,
                                       output=output,)
        
        return finished_req_outputs
    
    
    def _step_nonblocking(self):
        """Non-blocking step."""
        # 1. Schedule
        scheduler_output, req_ids = self._schedule()
        
        if self.engine_config.log_status:
            self._log_system_states(scheduler_output)

        # 2. Wait for result from previous round
        # This must be after step 1 to truly overlap scheduling and execution.
        prepare_output, denoising_output, postprocessing_output = (
            self.get_prev_handlers_output(get_output_all_workers=False))

        # 3. Schedule prepare if prepare reqs available
        if scheduler_output.has_prepare_requests():
            # We don't expect prepare stage if we have overlapped prepare-requests to process
            assert scheduler_output.status != RequestStatus.PREPARE
            self.prev_prepare_handlers = self._run_workers_nonblocking(
                "exec_prepare_stage", 
                self.prepare_workers,
                scheduler_reqs=scheduler_output.get_prepare_reqs_as_list(),
                use_mixed_precision=self.scheduler_config.use_mixed_precision)

        # 4. Issue tasks to workers
        if scheduler_output.status ==RequestStatus.EMPTY:
            # Empty indicates that no reqs to run, we don't need to do anything.
            if prepare_output is not None:
                # We don't need to preserve the handlers
                self._run_workers_nonblocking(
                    "receive_prepare_output",
                    self.workers,
                    prepare_output=prepare_output,
                )
        elif scheduler_output.status == RequestStatus.WAITING:
            # Currently, we don't do anything in waiting stage
            if prepare_output is not None:
                # We don't need to preserve the handlers
                self._run_workers_nonblocking(
                    "receive_prepare_output",
                    self.workers,
                    prepare_output=prepare_output,
                )
        elif scheduler_output.status == RequestStatus.PREPARE:
            # Only when there is no denoising or postprocessing reqs running will
            # prepare stage be scheduled.
            if prepare_output is not None:
                # We don't need to preserve the handlers
                self._run_workers_nonblocking(
                    "receive_prepare_output",
                    self.workers,
                    prepare_output=prepare_output,
                )
            # Requests are derived from normal reqs instead of prepare_reqs in shceduler_output
            self.prev_prepare_handlers = self._run_workers_nonblocking(
                "exec_prepare_stage",
                self.prepare_workers,
                scheduler_reqs=scheduler_output.get_reqs_as_list(),
                use_mixed_precision=self.scheduler_config.use_mixed_precision)
        elif scheduler_output.status == RequestStatus.DENOISING:
            # For denoising stage inference
            # transfer prepare result from previous round to worker
            self.prev_denoising_handlers = self._run_workers_nonblocking(
                "exec_denoising_stage", 
                self.workers,
                req_ids=req_ids,
                use_mixed_precision=self.scheduler_config.use_mixed_precision,
                is_sliced=scheduler_output.is_sliced,
                patch_size=scheduler_output.patch_size,
                prepare_output=prepare_output,)
        elif scheduler_output.status == RequestStatus.POSTPROCESSING:
            # For post stage inference
            self.prev_postprocessing_handlers = self._run_workers_nonblocking(
                "exec_post_stage",
                self.workers,
                req_ids=req_ids,
                use_mixed_precision=self.scheduler_config.use_mixed_precision,
                prepare_output=prepare_output,
            )
        else:
            raise RuntimeError(f"Unexpected status {str(scheduler_output.status)}.")
        
        # 5. Process output and update requests status.
        output = self._process_nonblocking_output(scheduler_output=scheduler_output,
                                                  req_ids=req_ids,
                                                  prepare_output=prepare_output,
                                                  denoising_output=denoising_output,
                                                  postprocessing_output=postprocessing_output,)
        
        return output

    
    def step(self) -> List[RequestOutput]:
        """One step consists of scheduling and execution of requests."""
        self._step_counter += 1
        if self.engine_config.non_blocking_step:
            return self._step_nonblocking()
        else:
            return self._step_blocking()
    
    
    def _process_nonblocking_output(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: List[int],
        prepare_output,
        denoising_output,
        postprocessing_output,
    ) -> List[RequestOutput]:
        # 1. update status of reqs in this round to new status to ensure consistency
        finished_reqs = self.scheduler.update_reqs_status_nonblocking(
            scheduler_output,
            req_ids,
            prepare_output,
            denoising_output,
            postprocessing_output,
            self.prev_scheduler_output,
        )

        # 2. Free finished requests.
        # This cannot be be done as blocking version, since `POSTPROCESSING` requests in
        # this round has already been updated to `FINISHED_STOPPED` status. We should refrain
        # from freeing them and only free those examined by scheduler and returned in step 1.
        self.scheduler.free_finished_requests(finished_reqs)

        # 3. Abort reqs
        if scheduler_output.abort_req_ids:
            finished_reqs.extend(self.abort_requests(scheduler_output.abort_req_ids))

        ret = []
        for req in finished_reqs:
            ret.append(RequestOutput(req))

        if self.collect_data and self.prev_scheduler_output:
            if self.prev_scheduler_output.status == RequestStatus.DENOISING:
                target_worker_output = denoising_output
            elif self.prev_scheduler_output.status == RequestStatus.POSTPROCESSING:
                target_worker_output = postprocessing_output
            else:
                target_worker_output = prepare_output
            self._collect_data(self.prev_scheduler_output, ret, target_worker_output)

        # Update
        self.prev_scheduler_output = scheduler_output

        return ret

    
    def _process_output(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: List[int],
        output: Optional[WorkerOutput],
    ) -> List[RequestOutput]:
        """Update requests status and prepare return result if available."""
        
        # Update the scheduled sequence groups with the model outputs
        self.scheduler.update_reqs_status(scheduler_output=scheduler_output,
                                          output=output,
                                          req_ids=req_ids)
        # collect finished reqs
        finished_reqs = self.scheduler.get_finished_requests()

        # abort reqs
        if scheduler_output.abort_req_ids:
            self.abort_requests(scheduler_output.abort_req_ids)

        # Create output wrappers
        ret = []
        for req in finished_reqs:
            ret.append(RequestOutput(req))
        
        if self.collect_data:
            self._collect_data(scheduler_output, ret, output)
        
        # free finished reqs
        self.scheduler.free_finished_requests(finished_reqs)
        
        return ret

    
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


    def _run_workers_nonblocking(
        self,
        method: str,
        workers: List[RayWorker],
        *args,
        **kwargs,
    ) -> List[ray.ObjectRef]:
        """Run designated workers in non-blocking form. Only ray workers
        support this function.

        Args:
            method (str): method name
            workers (List[RayWorker]): List of workers to run the method.

        Raises:
            RuntimeError: This exception will be raised if workers don't use ray.

        Returns:
            List[ray.ObjectRef]: List of object references of ray to get the results later.
        """
        if self.engine_config.log_status:
            logger.info(f"_run_workers_nonblocking start method {method}")
        assert self.parallel_config.worker_use_ray, "Only ray workers supports non blocking calls."
        obj_refs = []
        for worker in workers:
            obj_ref = worker.execute_method.remote(method, *args, **kwargs)
            obj_refs.append(obj_ref)
        return obj_refs
        
    
    def _run_workers_blocking(
        self,
        method: str,
        workers: List[RayWorker],
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
        if self.engine_config.log_status:
            logger.info(f"_run_workers_blocking start method {method}")
        all_outputs = []
        if max_concurrent_workers:
            work_groups = [
                workers[i:i + max_concurrent_workers]
                for i in range(0, len(self.workers), max_concurrent_workers)
            ]
        else:
            work_groups = [workers]

        for worker_subgroup in work_groups:
            all_outputs.extend(
                self._run_workers_in_batch(worker_subgroup, method, *args, **kwargs)
            )
        
        if get_all_outputs:
            return all_outputs
        else:
            output = all_outputs[0]
            for other_output in all_outputs[1:]:
                assert output == other_output, "Trying to ignore other valid outputs."
            return output
        
        
    def get_prev_handlers_output(
            self, 
            get_output_all_workers: bool = False
        ) -> Union[List[WorkerOutput], WorkerOutput]:
        """Get output from handlers set by previous round.

        Args:
            get_output_all_workers (bool, optional): If true, outputs from all workers
                will be returned as a list. Otherwise only the first output will be extracted
                and returned.
        """
        prepare_output = denoising_output = postprocessing_output = None
        if self.prev_prepare_handlers:
            prepare_output = ray.get(self.prev_prepare_handlers)
            self.prev_prepare_handlers = None
        if self.prev_denoising_handlers:
            denoising_output = ray.get(self.prev_denoising_handlers)
            self.prev_denoising_handlers = None
        if self.prev_postprocessing_handlers:
            postprocessing_output = ray.get(self.prev_postprocessing_handlers)
            self.prev_postprocessing_handlers = None

        if get_output_all_workers:
            return prepare_output, denoising_output, postprocessing_output

        prepare_output = prepare_output[0] if prepare_output else prepare_output
        denoising_output = denoising_output[0] if denoising_output else denoising_output
        postprocessing_output = postprocessing_output[0] if postprocessing_output else postprocessing_output
        return prepare_output, denoising_output, postprocessing_output

    
    def get_pipeline_config(self) -> PipelineConfig:
        """Gets the model configuration."""
        return self.pipeline_config


    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_normal_reqs()


    def has_unfinished_normal_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_normal_requests(is_nonblocking=self.engine_config.non_blocking_step)

    
    def _log_system_states(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # record_metrics(
        #     avg_prompt_throughput=avg_prompt_throughput,
        #     avg_generation_throughput=avg_generation_throughput,
        #     scheduler_running=len(self.scheduler.running),
        #     scheduler_swapped=len(self.scheduler.swapped),
        #     scheduler_waiting=len(self.scheduler.waiting),
        #     gpu_cache_usage=gpu_cache_usage,
        #     cpu_cache_usage=cpu_cache_usage,
        # ) 

        self.scheduler.log_status()
        logger.info(scheduler_output.get_log_string())
        sys.stdout.flush()
    

    def clear(self):
        if self.collect_data:
            self.sm_monitor.end_monitor()
            self.sm_monitor.log_result_to_file()
    
    
    def _collect_data(
        self,
        scheduler_output: 'SchedulerOutput',
        req_outputs: List[RequestOutput],
        worker_output: WorkerOutput,
    ) -> None:
        self.sm_monitor.checkpoint()
        # Request data
        for ro in req_outputs:
            self.request_logger.info(
                f"{ro.request_id},{ro.normal_finished},{ro.start_datetime},{ro.finish_datetime},"
                f"{ro.time_consumption}"
            )
        # Schedule data
        res_req_num = []
        total = 0
        for res in self.support_resolutions:
            if res in scheduler_output.scheduled_requests:
                num = len(scheduler_output.scheduled_requests[res])
            else:
                num = 0
            res_req_num.append(num)
            total += num
        worker_step_time = (worker_output.end_time - worker_output.start_time) if worker_output is not None else None
        self.schedule_logger.info(
            f"{self._step_counter},{datetime.datetime.now()},{str(scheduler_output.status)},{total},"
            f"{res_req_num[0]},{res_req_num[1]},"
            f"{res_req_num[2]},{worker_step_time}"
        )
    
    
    def _prepare_collect_data(self):
        model = os.getenv("MODEL")
        distribution = os.getenv("DISTRIBUTION")
        qps = os.getenv("QPS")
        slo = os.getenv("SLO")
        policy = os.getenv("POLICY")

        result_dir_path = f"./results/{model}/{distribution}_{qps}_{slo}_{policy}"
        os.makedirs(result_dir_path + "/imgs", exist_ok=True)

        sm_util_file_name = result_dir_path + "/sm_util.csv"
        request_data_file_name = result_dir_path + "/request_data.csv"
        schedule_data_file_name = result_dir_path + "/schedule.csv"

        # Get loggers
        def prepare_logger(filename: str):
            local_logger = logging.getLogger(filename)
            local_logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(filename, mode="w")
            local_logger.addHandler(handler)
            local_logger.propagate = False
            return local_logger

        self.request_logger = prepare_logger(request_data_file_name)
        self.request_logger.info("request_id,is_finished,start_time,finish_time,time_consumption")
        self.schedule_logger = prepare_logger(schedule_data_file_name)
        self.schedule_logger.info(
            f"step_count,timestamp,status,num_scheduled_reqs,num_req_of_{self.support_resolutions[0]},"
            f"num_req_of_{self.support_resolutions[1]},num_req_of_{self.support_resolutions[2]},"
            f"step_worker_time_consumption"
        )

        
        self.sm_monitor = SmUtilMonitor(sm_util_file_name, interval=0.1)
        self.sm_monitor.start_monitor()