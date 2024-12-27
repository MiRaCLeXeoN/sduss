import copy
import os
import datetime
import sys
import logging
import traceback

from typing import Optional, Union, List, Any, Tuple, Dict, TYPE_CHECKING, Iterable
from functools import partial

from sduss.dispatcher import Dispatcher, Request, DispatcherResultType
from sduss.worker import WorkerOutput
from sduss.logger import init_logger
from sduss.entrypoints.wrappers import ReqOutput
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.config import (PipelineConfig, ParallelConfig, SchedulerConfig, EngineConfig)
from sduss.engine.arg_utils import EngineArgs
from sduss.executor.mp_executor import mp_init_method
from sduss.executor.mp_executor import MpExecutor
from sduss.engine.metrics import record_metrics

from .utils import SmUtilMonitor

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup
    from sduss.model_executor.diffusers import BasePipeline

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5

def worker_init_fn(
        model_config, parallel_config, scheduler_config, engine_config, 
        rank, device, is_prepare_worker=False, distributed_init_method = None,
    ):
    from sduss.worker.worker import Worker
    return Worker(model_config, parallel_config, scheduler_config, 
                    engine_config, rank=rank, device=device, is_prepare_worker=is_prepare_worker,
                    distributed_init_method=distributed_init_method)

class Engine:
    """The main engine that receives requests and generates texts.
    """
    
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        engine_config: EngineConfig,
        **kwargs,
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
        self._init_workers_mp(mp_init_method())
        if engine_config.log_status:
            logger.info("All workers initialization complete")
            
        self.pipeline_cls = None
        self.sampling_param_cls = None
        self._set_pipeline_cls()
        
        self.support_resolutions = self.pipeline_cls.SUPPORT_RESOLUTIONS
        self.dispatcher = Dispatcher(scheduler_config, parallel_config, engine_config, self.support_resolutions)
        if engine_config.log_status:
            logger.info("Dispatcher initialization complete")

        self._step_counter = 0
        self.engine_ready = True
        # Flust outputs so that we can see logs ASAP.
        if engine_config.log_status:
            logger.info("Engine initialization done. System ready.")
        
        # FIXME: For experiment
        self.collect_data = os.getenv("SDUSS_COLLECT_DATA")
        if self.collect_data:
            self._prepare_collect_data()


    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "Engine":
        """Create an inference engine from arguments"""
        # Create engine configs.
        pipeline_config, parallel_config, scheduler_config, engine_config= engine_args.create_engine_configs()

        # Create engine instance
        return cls(pipeline_config, 
                   parallel_config, 
                   scheduler_config, 
                   engine_config,)
    

    def _set_pipeline_cls(self) -> None:
        from sduss.model_executor.model_loader import get_pipeline_cls
        self.pipeline_cls: BasePipeline = get_pipeline_cls(self.pipeline_config)
        self.sampling_param_cls = self.pipeline_cls.get_sampling_params_cls()


    def _verify_args(self):
        """Verify args. Now only parallel config requires verification."""
        self.parallel_config.verify_with_scheduler_config(self.scheduler_config)
        self.pipeline_config.verify_with_scheduler_config(self.scheduler_config)
        self.engine_config.verify_with_scheduler_config(self.scheduler_config)

    
    def _init_workers_mp(
        self,
        distributed_init_method: str,
    ):
        self.workers: 'List[MpExecutor]' = []
        for i in range(self.parallel_config.num_gpu_workers):
            worker = MpExecutor(f"sduss_gpu_worker{i}", rank=i, device=self.parallel_config.gpus[i], is_prepare_worker=False)
            self.workers.append(worker)

        self.prepare_workers: 'List[MpExecutor]' = []
        for i in range(self.parallel_config.num_cpu_workers):
            worker = MpExecutor(f"sduss_cpu_worker{i}", rank=i, device=self.parallel_config.gpus[i], is_prepare_worker=True)
            self.prepare_workers.append(worker)
        
        # init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.pipeline_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        engine_config = copy.deepcopy(self.engine_config)
        
        # execute `init_worker` 
        for i, worker in enumerate(self.workers):
            worker.init_worker(worker_init_fn=partial(worker_init_fn, model_config, parallel_config, 
                                                      scheduler_config, engine_config, rank=i, 
                                                      device=worker.device, is_prepare_worker=False,
                                                      distributed_init_method=distributed_init_method))
        self._run_workers_blocking("init_dis_env", self.workers, get_all_outputs=True)
        if self.scheduler_config.overlap_prepare:
            for i, worker in enumerate(self.prepare_workers):
                worker.init_worker(worker_init_fn=partial(worker_init_fn, model_config, parallel_config, 
                                                            scheduler_config, engine_config, rank=i, device=-1, 
                                                            is_prepare_worker=True,
                                                            distributed_init_method=distributed_init_method))
            self._run_workers_blocking("init_prepare", self.prepare_workers, get_all_outputs=True)
        
        # Load model
        if self.scheduler_config.overlap_prepare:
            self._run_workers_blocking("load_model", self.workers + self.prepare_workers, get_all_outputs=True)
        else:
            self._run_workers_blocking("load_model", self.workers, get_all_outputs=True)


    def add_requests(
        self,
        new_requests_params: List[Dict],
    ) -> None:
        """Add a batch of requests."""
        reqs = []
        for req_param_dict in new_requests_params:
            req = Request(**req_param_dict)
            reqs.append[req]
        self.dispatcher.add_requests(reqs)

    
    def abort_requests(self, request_ids: Union[int, Iterable[int]]) -> List[Request]:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        return self.dispatcher.abort_requests(request_ids)

    
    def step(self) -> List[ReqOutput]:
        """One step consists of scheduling and execution of requests."""
        self._step_counter += 1
        try:
            reqs_by_dp: Dict[int, Request] = self.dispatcher.dispatch()
        except Exception as e:
            print(e)
            traceback.print_exc()
    
    
    def dispatch_reqs(self, reqs_by_dp: DispatcherResultType) -> None:
        
    
    
    def _process_nonblocking_output(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: List[int],
        prepare_output,
        denoising_output,
        postprocessing_output,
        overlapped_prepare_output,
    ) -> List[ReqOutput]:
        # 1. update status of reqs in this round to new status to ensure consistency
        finished_reqs = self.dispatcher.update_reqs_status_nonblocking(
            scheduler_output,
            req_ids,
            prepare_output,
            denoising_output,
            postprocessing_output,
            overlapped_prepare_output,
            self.prev_scheduler_output,
            overlapped_prepare_sche_opt=None if not self.overlapped_prepare_sche_opt else self.overlapped_prepare_sche_opt[0],
        )


        # 2. Free finished requests.
        # This cannot be be done as blocking version, since `POSTPROCESSING` requests in
        # this round has already been updated to `FINISHED_STOPPED` status. We should refrain
        # from freeing them and only free those examined by scheduler and returned in step 1.
        self.dispatcher.free_finished_requests(finished_reqs)

        # 3. Abort reqs
        if scheduler_output.abort_req_ids:
            finished_reqs.extend(self.abort_requests(scheduler_output.abort_req_ids))

        ret = []
        for req in finished_reqs:
            ret.append(ReqOutput(req))

        if self.collect_data and self.prev_scheduler_output:
            if self.prev_scheduler_output.status == ReqStatus.DENOISING:
                target_worker_output = denoising_output
            elif self.prev_scheduler_output.status == ReqStatus.POSTPROCESSING:
                target_worker_output = postprocessing_output
            else:
                target_worker_output = prepare_output
            self._collect_data(self.prev_scheduler_output, ret, target_worker_output)

        # Update
        self.prev_scheduler_output = scheduler_output
        if overlapped_prepare_output is not None:
            # Reset after use
            self.overlapped_prepare_sche_opt.pop(0)


        return ret
    
    
    def _get_output_nonblocking(
        self,
        handlers,
    ) -> Optional['WorkerOutput']:
        outputs = []
        if self.parallel_config.worker_use_mp:
            for worker in handlers:
                if worker.data_is_available():
                    outputs.append(worker.get_blocking())

        # Currently, we only suppose 1 worker to get result from
        if not outputs:
            return None

        assert len(outputs) == 1
        return outputs[0]

    
    def _process_output(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: List[int],
        output: Optional[WorkerOutput],
    ) -> List[ReqOutput]:
        """Update requests status and prepare return result if available."""
        
        # Update the scheduled sequence groups with the model outputs
        self.dispatcher.update_reqs_status(scheduler_output=scheduler_output,
                                          output=output,
                                          req_ids=req_ids)
        # collect finished reqs
        finished_reqs = self.dispatcher.get_finished_requests()

        # abort reqs
        if scheduler_output.abort_req_ids:
            self.abort_requests(scheduler_output.abort_req_ids)

        # Create output wrappers
        ret = []
        for req in finished_reqs:
            ret.append(ReqOutput(req))
        
        if self.collect_data:
            self._collect_data(scheduler_output, ret, output)
        
        # free finished reqs
        self.dispatcher.free_finished_requests(finished_reqs)
        
        return ret


    def _run_workers_nonblocking(
        self,
        method: str,
        workers: List,
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
        obj_refs = []
        if self.parallel_config.worker_use_ray:
            for worker in workers:
                obj_ref = worker.execute_method.remote(method, *args, **kwargs)
                obj_refs.append(obj_ref)
        elif self.parallel_config.worker_use_mp:
            for worker in workers:
                obj_ref = worker.execute_method(method, *args, **kwargs)
                obj_refs.append(obj_ref)
        else:
            raise RuntimeError("Your chosen worker type doesn't support nonblokcing method at now.")
        return obj_refs
        

    def _run_workers_in_batch(
        self,
        workers: List[MpExecutor],
        method: str,
        *args,
        **kwargs,
    ):
        all_outputs = []
        # 1. Add task to each worker
        # We must add tasks to all workers before waiting for any of them!
        for worker in workers:
            output = worker.execute_method(method, *args, **kwargs)
            all_outputs.append(output)
        
        all_outputs = [worker.get_blocking() for worker in all_outputs]
        return all_outputs
        

    def _run_workers_blocking(
        self,
        method: str,
        workers: List[MpExecutor],
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
            logger.info(f"_run_workers_blocking start method {method}, {get_all_outputs=}")
        
        # If no workers specified, directly return
        if not workers:
            return

        # Split workers into subgroups
        all_outputs = []
        if max_concurrent_workers:
            work_groups = [
                workers[i:i + max_concurrent_workers]
                for i in range(0, len(self.workers), max_concurrent_workers)
            ]
        else:
            work_groups = [workers]

        # Launch tasks subgroup by subgroup
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
        
        
    
    def get_pipeline_config(self) -> PipelineConfig:
        """Gets the model configuration."""
        return self.pipeline_config


    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.dispatcher.get_num_unfinished_normal_reqs()


    def has_unfinished_normal_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.dispatcher.has_unfinished_normal_requests(is_nonblocking=self.engine_config.non_blocking_step)

    
    def _log_system_states(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        self.dispatcher.log_status()
        logger.info(scheduler_output.get_log_string())
        sys.stdout.flush()
    


    def engine_is_ready(self) -> bool:
        return self.engine_ready
    
    

    def clear(self):
        if self.collect_data:
            self.sm_monitor.end_monitor()
            self.sm_monitor.log_result_to_file()
        sys.stdout.flush()
        sys.stderr.flush()
    
    
    def _collect_data(
        self,
        scheduler_output: 'SchedulerOutput',
        req_outputs: List[ReqOutput],
        worker_output: WorkerOutput,
    ) -> None:
        self.sm_monitor.checkpoint()
        # Request data
        for ro in req_outputs:
            self.request_logger.info(
                f"{ro.request_id},{ro.normal_finished},{ro.start_datetime},{ro.finish_datetime},"
                f"{ro.resolution},{ro.time_consumption}"
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
            f"{res_req_num[2]},{len(scheduler_output.get_prepare_reqs_as_list())},{worker_step_time}"
        )
    
    
    def _prepare_collect_data(self):
        model = os.getenv("MODEL")
        distribution = os.getenv("DISTRIBUTION")
        qps = os.getenv("QPS")
        slo = os.getenv("SLO")
        policy = os.getenv("POLICY")
        arrival_distri = os.getenv("ARRIVAL_DISTRI")
        result_dir_path = f"./results/{model}/{arrival_distri}/{distribution}_{qps}_{slo}_{policy}"
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
        self.request_logger.info("request_id,is_finished,start_time,finish_time,resolution,time_consumption")
        self.schedule_logger = prepare_logger(schedule_data_file_name)
        self.schedule_logger.info(
            f"step_count,timestamp,status,num_scheduled_reqs,num_req_of_{self.support_resolutions[0]},"
            f"num_req_of_{self.support_resolutions[1]},num_req_of_{self.support_resolutions[2]},"
            f"num_req_of_overlap_prepare,"
            f"step_worker_time_consumption"
        )

        
        self.sm_monitor = SmUtilMonitor(sm_util_file_name, interval=0.01)
        self.sm_monitor.start_monitor()