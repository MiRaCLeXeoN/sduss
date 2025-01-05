import time
import torch

from typing import Optional, List, Dict, Union, TYPE_CHECKING, Any, Tuple

from sduss.config import PipelineConfig, ParallelConfig, SchedulerConfig, EngineConfig
from sduss.model_executor import get_pipeline_cls
from sduss.logger import init_logger

from .scheduler.scheduler import Scheduler, SchedulerOutput
from .runner.model_runner import ModelRunner
from .wrappers import WorkerOutput, WorkerRequest, WorkerReqStatus

if TYPE_CHECKING:
    from .runner.wrappers import RunnerOutput

logger = None

class Worker:
    """A worker GPU class
    
    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of distributed
    inference, each worker is assigned a partition of the model
    """
    
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        engine_config: EngineConfig,
        rank: int,
        device: int,
        is_prepare_worker: bool = False,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_config (ModelConfig): Model config
            parallel_config (ParallelConfig): Parallel config
            scheduler_config (SchedulerConfig): Scheduler config
            rank (Optional[int], optional): _description_. Defaults to None.
            distributed_init_method (Optional[str], optional): _description_. Defaults to None.
        """
        self.pipeline_config = pipeline_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.engine_config = engine_config
        self.rank = rank
        self.device_num = device
        self.is_prepare_worker = is_prepare_worker
        self.distributed_init_method = distributed_init_method

        self.use_esymred = pipeline_config.use_esymred
        self.use_mixed_precision = scheduler_config.use_mixed_precision


        self.pipeline_cls = get_pipeline_cls(self.pipeline_config)
        self.model_runner = ModelRunner(pipeline_config, parallel_config, 
                                        scheduler_config, is_prepare_worker,
                                        name=f"ModelRunner rank {self.rank}",
                                        rank=rank,
                                        device_num=device,
                                        distributed_init_method=distributed_init_method,)
        self.scheduler = Scheduler(self.scheduler_config, 
                                   self.engine_config, 
                                   self.pipeline_cls.SUPPORT_RESOLUTIONS)


        # Set afterwards
        self.device = None
        self.prev_task = None
        self.prev_sche_output = None

        global logger
        logger = init_logger(__name__, no_stdout=True, to_file_name=f"./outputs/gpu_worker_{self.rank}.log")
    

    def init_dis_env(self) -> None:
        self.model_runner.execute_method_sync("init_dis_env")
    
    
    def init_prepare(self) -> None:
        self.model_runner.execute_method_sync("init_prepare")
        
    
    def load_model(self) -> None:
        self.model_runner.execute_method_sync("load_model")
                                
    
    def step(self) -> Tuple[WorkerOutput, bool]:
        # 1. Schedule
        ## ! We must check unfinished reqs before reqs get updated by this round
        ## ! Since in unblocking fashion, when the very last req is running post stage,
        ## ! it will be updated to finished status. So has_unfinished_reqs will be 0, even if
        ## ! one is still running.
        has_unfinished_reqs = self.scheduler.has_unfinished_requests()
        t1 = time.time()
        if has_unfinished_reqs:
            scheduler_output = self.scheduler.schedule()
        else:
            # We have last reqs running post stage, no need to schedule
            scheduler_output = SchedulerOutput(status=WorkerReqStatus.EMPTY)
        t2 = time.time()
        
        # 2. Issue task of this round
        # Issue task before getting result, overlapping more regions.
        task = self._issue_task(scheduler_output)
        t3 = time.time()

        # 3. Wait for results from previous round
        prev_output = None
        if self.prev_task:
            prev_output = self.model_runner.get_result(self.prev_task)
        t4 = time.time()

        # Log
        if self.engine_config.log_status:
            self.log_status(
                schedule_time=t2 - t1,
                get_result_time=t4 - t3,
                scheduler_output=scheduler_output,
                prev_output=prev_output,
            )
        
        # 4. Process output and update requests status
        if scheduler_output.abort_req_ids is not None:
            aborted_reqs = self.abort_requests(scheduler_output.abort_req_ids)
        else:
            aborted_reqs = []
        self._update_reqs(scheduler_output)
        ## finished reqs are automatically removed from request pool by scheduler
        finished_reqs = self._process_prev_output(prev_output, self.prev_sche_output)
        
        self.prev_task = task
        self.prev_sche_output = scheduler_output

        if len(finished_reqs) == 0 and len(aborted_reqs) == 0:
            return None, has_unfinished_reqs

        worker_output = WorkerOutput(worker_reqs=finished_reqs, aborted_reqs=aborted_reqs)
        return worker_output, has_unfinished_reqs
    
    
    def _issue_task(self, scheduler_output: 'SchedulerOutput') -> None:
        status = scheduler_output.status
        if status == WorkerReqStatus.EMPTY:
            # Nothing to do
            task = None
        elif status == WorkerReqStatus.PREPARE:
            reqs = scheduler_output.get_reqs_as_list()
            req_ids = []
            req_sps = []
            for req in reqs:
                req_ids.append(req.request_id)
                req_sps.append(req.sampling_params)
            task = self.model_runner.execute_method_async("exec_prepare_stage", need_res=True, 
                                                          req_ids=req_ids, req_sps=req_sps)
        elif status == WorkerReqStatus.DENOISING:
            task = self.model_runner.execute_method_async("exec_denoising_stage", need_res=True,
                                                         req_ids=scheduler_output.get_req_ids(),
                                                         is_sliced=scheduler_output.is_sliced,
                                                         patch_size=scheduler_output.patch_size,)
        elif status == WorkerReqStatus.POSTPROCESSING:
            task = self.model_runner.execute_method_async("exec_post_stage", need_res=True,
                                                          req_ids=scheduler_output.get_req_ids(),)
        else:
            raise RuntimeError(f"Unexpected {status=} from scheduler.")
        
        return task
    
    
    def _process_prev_output(self, prev_output: 'RunnerOutput', 
                             prev_sche_output: 'SchedulerOutput') -> List[WorkerRequest]:
        if not prev_output:
            # First round, no output
            return []
        return self.scheduler.process_output(prev_sche_output, prev_output)
        
    
    def _update_reqs(self, scheduler_output: 'SchedulerOutput'):
        # 1. update status of reqs in this round to new status to ensure consistency
        self.scheduler.update_reqs_status(scheduler_output)
    

    def add_requests(self, req_ids: List[int], req_sps: List[Any]):
        reqs = []
        for req_id, req_sp in zip(req_ids, req_sps):
            req = WorkerRequest(request_id=req_id, sampling_params=req_sp)
            reqs.append(req)
        self.scheduler.add_requests(reqs)
    
    
    def abort_requests(self, req_ids: Union[int, List[int]]):
        if isinstance(req_ids, int):
            req_ids = [req_ids]
        return self.scheduler.abort_requests(req_ids)
    

    def log_status(self, schedule_time, get_result_time, scheduler_output: 'SchedulerOutput', prev_output: 'RunnerOutput'):
        logger.debug(self.scheduler.get_log_status_str())
        logger.debug(f"{schedule_time=}, {get_result_time=}")
        if self.prev_sche_output and prev_output:
            logger.debug(f"prev schedule status: {self.prev_sche_output.status}, prev exec time: {prev_output.end_time - prev_output.start_time}s")
    
    
    def shutdown(self) -> None:
        self.model_runner.shutdown()