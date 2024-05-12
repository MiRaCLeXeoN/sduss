import argparse

from dataclasses import dataclass, fields
from typing import Optional, Tuple

from sduss.config import (PipelineConfig, ParallelConfig, SchedulerConfig,
                          EngineConfig)
class EngineArgs:
    """Arguments for the base class Engine
    """
    def __init__(self, model_name_or_pth: str, **kwargs) -> None:
        # Model configs
        self.model_name_or_pth = model_name_or_pth
        self.trust_remote_code = kwargs.pop("trust_remote_code", False)
        self.seed = kwargs.pop("seed", 10086)
        self.use_esymred = kwargs.pop("use_esymred", False)
        self.use_batch_split = kwargs.pop("use_batch_split", False)
        # Distributed configs
        self.worker_use_ray = kwargs.pop("worker_use_ray", True)
        self.pipeline_parallel_size = kwargs.pop("pipeline_parallel_size", 1)
        self.tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        self.data_parallel_size = kwargs.pop("data_parallel_size", 1)
        self.num_cpus_extra_worker = kwargs.pop("num_cpus_extra_worker ", 2)
        self.max_parallel_loading_workers = kwargs.pop("max_parallel_loading_workers", None)
        # Scheduler configs
        self.max_batchsize = kwargs.pop("max_batchsize", 32)
        self.use_mixed_precisoin = kwargs.pop("use_mixed_precision", False)
        self.policy = kwargs.pop("policy", "fcfs_single")
        self.overlap_prepare = kwargs.pop("overlap_prepare", False)
        # Engine configs
        self.disable_log_status = kwargs.pop("disable_log_status", False)
        self.non_blocking_step = kwargs.pop("non_blocking_step", False)
        # kwargs for `from_pretrained`
        self.kwargs = kwargs


    @staticmethod
    def add_args_to_parser(
        parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        # NOTE: If you update any of the arguments below, please also
        # make sure to update docs/source/models/engine_args.rst
        pass
    

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # get all the attributes into the form of list
        raise NotImplementedError

    
    def get_pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(
            pipeline=self.model_name_or_pth, 
            trust_remote_code=self.trust_remote_code,
            seed=self.seed, 
            use_esymred=self.use_esymred,
            use_batch_split=self.use_batch_split,
            kwargs=self.kwargs
        )
    
    def get_parallel_config(self) -> ParallelConfig
        return ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            data_parallel_size=self.data_parallel_size,
            num_cpus_extra_worker=self.num_cpus_extra_worker,
            worker_use_ray=self.worker_use_ray,
            max_parallel_loading_workers=self.max_parallel_loading_workers
        )
    
    def get_scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            max_bathsize=self.max_batchsize,
            use_mixed_precision=self.use_mixed_precisoin,
            policy=self.policy,
            overlap_prepare=self.overlap_prepare,
        )
    
    def get_engine_config(self) -> EngineConfig:
        return EngineConfig(
            log_status=not self.disable_log_status,
            non_blocking_step=self.non_blocking_step
        )
        
    
    def create_engine_configs(
        self,
    ) -> Tuple[PipelineConfig, ParallelConfig, SchedulerConfig, EngineConfig]:
        pipeline_config = self.get_engine_config()
        parallel_config = self.get_scheduler_config()
        scheduler_config = self.get_parallel_config()
        engine_config = self.get_pipeline_config()
        return pipeline_config, parallel_config, scheduler_config, engine_config


class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous engine, inherited from EngineArgs """
    def __init__(self, model_name_or_pth: str, **kwargs) -> None:
        self.engine_use_ray = kwargs.pop("engine_use_ray", False)
        self.disable_log_requests = kwargs.pop("disable_log_requests", False)
        self.max_log_len = kwargs.pop("max_log_len", None)
        super().__init__(model_name_or_pth, **kwargs)
    
    
    def get_engine_config(self) -> EngineConfig:
        return EngineConfig(
            log_status=not self.disable_log_status,
            non_blocking_step=self.non_blocking_step,
            engine_use_ray=self.engine_use_ray,
            log_requests=not self.disable_log_requests,
        )
    
    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # add args from base engine
        parser = EngineArgs.add_args_to_parser(parser)
        
        parser.add_argument(
            '--engine-use-ray',
            action='store_true',
            help='Use Ray to start the Execution engine in a separate process '
                'as the server process'
        )
        parser.add_argument(
            '--disable-log-requests',
            action='store_true',
            help='Disable logging requests\' timeline'
        )
        
        return parser