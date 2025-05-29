import copy
import os
import datetime
import sys
import logging
import traceback
import asyncio

from typing import Optional, Union, List, Any, Tuple, Dict, TYPE_CHECKING, Iterable
from functools import partial

from sduss.dispatcher import Dispatcher, Request, DispatcherResultType
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
    from sduss.worker import WorkerOutput

logger = init_logger(__name__)


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

        # init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.pipeline_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        engine_config = copy.deepcopy(self.engine_config)
        
        # execute `init_worker` 
        for i, worker in enumerate(self.workers):
            worker.init_worker(worker_init_fn=partial(worker_init_fn, model_config, parallel_config, 
                                                      scheduler_config, engine_config, rank=i, 
                                                      device=worker.device_num, is_prepare_worker=False,
                                                      distributed_init_method=distributed_init_method))
        method_dict = {w:([], {}) for i, w in enumerate(self.workers)}
        self._run_workers("init_dis_env", method_dict, blocking=True)

        
        # Load model
        method_dict = {w:([], {}) for i, w in enumerate(self.workers)}
        self._run_workers("load_model", method_dict, blocking=True)


    def add_requests(
        self,
        new_requests_params: List[Dict],
    ) -> None:
        """Add a batch of requests."""
        reqs = []
        for req_param_dict in new_requests_params:
            req = Request(**req_param_dict)
            reqs.append(req)
        self.dispatcher.add_requests(reqs)

    
    def abort_requests(self, request_ids: Union[int, Iterable[int]]) -> List[Request]:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        return self.dispatcher.abort_requests(request_ids)

    
    def step(self) -> Tuple[List[ReqOutput], bool]:
        """One step consists of scheduling and execution of requests."""
        self._step_counter += 1
        try:
            reqs_by_dp: Dict[int, Request] = self.dispatcher.dispatch()
            if len(reqs_by_dp) > 0:
                self._dispatch_reqs(reqs_by_dp)

            worker_outputs = self._get_outputs_nowait()
            req_outputs = self._process_outputs(worker_outputs)
            if req_outputs and self.collect_data:
                self._collect_data(req_outputs)
            return req_outputs, self.has_unfinished_requests()
        except Exception as e:
            print(e)
            traceback.print_exc()
    
    
    def _dispatch_reqs(self, reqs_by_dp: DispatcherResultType) -> None:
        # 1. build method dict
        method_dict = {}
        for dp_rank in reqs_by_dp.keys():
            method_dict[self.workers[dp_rank]] = ([], 
            {
                "req_ids" : [req.request_id for req in reqs_by_dp[dp_rank]],
                "req_sps" : [req.sampling_params for req in reqs_by_dp[dp_rank]],
            })
        # 2. launch method
        self._run_workers("add_requests", method_dict, blocking=False, need_res=False)
        return None
    
    
    def _get_outputs_nowait(
        self,
    ) -> 'List[WorkerOutput]':
        """Collect all outputs that are currently available.

        Returns:
            List[WorkerOutput]: _description_
        """
        worker_outputs = []
        for worker in self.workers:
            worker_outputs.extend(worker.get_output_nowait())

        return worker_outputs

    
    def _process_outputs(
        self,
        worker_outputs: 'List[WorkerOutput]',
    ) -> List[ReqOutput]:
        """Update requests status and prepare return result if available."""
        if len(worker_outputs) == 0:
            return []
        finished_reqs, aborted_reqs = self.dispatcher.process_worker_outputs(worker_outputs)
        return [ReqOutput(req) for req in finished_reqs + aborted_reqs]


    def _wait_handlers(self, handlers_dict: 'Dict[MpExecutor, asyncio.Task]') -> 'Dict[MpExecutor, Any]':
        handlers = []
        workers = []
        for w, h in handlers_dict.items():
            workers.append(w)
            handlers.append(h)

        outputs = asyncio.get_event_loop().run_until_complete(asyncio.gather(*handlers))

        output_dict = {}
        for i, o in enumerate(outputs):
            output_dict[workers[i]] = o

        return output_dict
        

    def _run_workers_nonblocking(
        self,
        method: str,
        workers_dict: 'Dict[MpExecutor, Tuple]',
        need_res: bool,
    ) -> 'Optional[Dict[MpExecutor, asyncio.Task]]':
        """Run workers in nonblocking fashion. Only handlers are returned.

        Args:
            method (str): method name
            workers_dict (Dict[MpExecutor, Tuple]): The argument dict for each worker
            need_res (bool): if set false, no handlers will be returned

        Returns:
            List[asyncio.Task]: Handlers for results retrieval
        """
        handlers_dict = {}
        for worker in workers_dict.keys():
            args, kwargs = workers_dict[worker]
            h = worker.execute_method_async(method, need_res, *args, **kwargs)
            handlers_dict[worker] = h

        if need_res:
            return handlers_dict
        else:
            return None


    def _run_workers(
        self,
        method: str,
        workers_dict: 'Dict[MpExecutor, Tuple]',
        blocking: bool,
        need_res: bool = True,
    ) -> 'Optional[Dict[MpExecutor, Any]]':
        """Run method on selected workers

        Args:
            method (str): method name
            workers_dict (Dict[MpExecutor, Tuple]): The structured input for each worker
            blocking (bool, optional): If true, block until the result is returned;
                If false, return handlers for results retrieval, in nonblocking fashion. 
                Defaults to True.

        Returns:
            Dict[MpExecutor, Any]: _description_
        """
        if self.engine_config.log_status:
            logger.info(f"_run_workers start method {method}, {blocking=}")
        
        # if we want the blocking method, we have to force need_res
        if blocking:
            need_res = True

        handlers_dict = self._run_workers_nonblocking(method, workers_dict, need_res)

        if not need_res:
            return None

        if blocking:
            output_dict = self._wait_handlers(handlers_dict)
            return output_dict
        else:
            return handlers_dict


    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.dispatcher.has_unfinished_reqs()


    def engine_is_ready(self) -> bool:
        return self.engine_ready
    

    def clear(self):
        sys.stdout.flush()
        sys.stderr.flush()

        for w in self.workers:
            w.end_worker()
        
    
    def _collect_data(
        self,
        req_outputs: List[ReqOutput],
    ) -> None:
        # Request data
        for ro in req_outputs:
            self.request_logger.info(
                f"{ro.request_id},{ro.normal_finished},{ro.start_datetime},{ro.finish_datetime},{ro.worker_arrival_time},{ro.worker_finish_time},"
                f"{ro.resolution},{ro.time_consumption},{ro.worker_time_consumption}"
            )
        num_reqs_by_dp_rank = self.dispatcher.get_num_unfinished_reqs_by_dp_rank()
        total = sum(list(num_reqs_by_dp_rank.values()))
        # Dispatch data
        row = f"{self._step_counter},{datetime.datetime.now()},{total},"
        row += ",".join([str(num_reqs_by_dp_rank[i]) for i in range(len(self.workers))])
        self.dispatch_logger.info(row)
    
    
    def _prepare_collect_data(self):
        model = os.getenv("MODEL")
        qps = os.getenv("QPS")
        slo = os.getenv("SLO")
        policy = os.getenv("POLICY")
        data_parallel_size = self.parallel_config.data_parallel_size
        result_dir_path = f"./results/{model}/{qps}_{slo}_{policy}_{data_parallel_size}"
        os.makedirs(result_dir_path, exist_ok=True)

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

        # Req data
        self.request_logger = prepare_logger(request_data_file_name)
        self.request_logger.info("request_id,normal_finished,start_time,finish_time,worker_start_time,worker_finish_time,resolution,time_consumption,worker_time_consumption")
        # Dispatch data
        self.dispatch_logger = prepare_logger(schedule_data_file_name)
        header = f"step_count,timestamp,total_running_reqs,"
        header += ",".join([f"num_worker{i}_reqs" for i in range(len(self.workers))])
        self.dispatch_logger.info(header)