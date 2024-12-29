import argparse

import torch

from dataclasses import dataclass, fields
from typing import Optional, Tuple

from sduss.config import (PipelineConfig, ParallelConfig, SchedulerConfig,
                          EngineConfig)

from .utils import get_torch_dtype_from_string, parse_ranges

class EngineArgs:
    """Arguments for the base class Engine
    """
    def __init__(self, **kwargs) -> None:
        # Model configs
        self.model_name_or_pth = kwargs.pop("model_name_or_pth", None)
        self.trust_remote_code = kwargs.pop("trust_remote_code", False)
        self.seed = kwargs.pop("seed", 10086)
        self.use_esymred = kwargs.pop("use_esymred", False)
        self.use_batch_split = kwargs.pop("use_batch_split", False)
        # Parallel configs
        self.worker_use_ray = kwargs.pop("worker_use_ray", False)
        self.worker_use_mp = kwargs.pop("worker_use_mp", True)
        self.pipeline_parallel_size = kwargs.pop("pipeline_parallel_size", 1)
        self.tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        self.data_parallel_size = kwargs.pop("data_parallel_size", 1)
        self.num_cpus_cpu_worker = kwargs.pop("num_cpus_cpu_worker", 8)
        self.num_cpus_gpu_worker = kwargs.pop("num_cpus_gpu_worker", 4)
        self.num_cpu_workers = kwargs.pop("num_cpu_workers", 1)
        self.gpus = parse_ranges(kwargs.pop("gpus", "[0]"))
        # Scheduler configs
        self.max_batchsize = kwargs.pop("max_batchsize", 32)
        self.use_mixed_precisoin = kwargs.pop("use_mixed_precision", False)
        self.policy = kwargs.pop("policy", "fcfs_single")
        self.overlap_prepare = kwargs.pop("overlap_prepare", False)
        # Engine configs
        self.disable_log_status = kwargs.pop("disable_log_status", False)
        self.dispatcher_policy = kwargs.pop("dispatcher_policy", "greedy")
        # kwargs for `from_pretrained`
        self.kwargs = kwargs

        assert self.model_name_or_pth is not None

        # NOTE: If you update any of the arguments above, please also
        # update argparse below.


    @staticmethod
    def add_args_to_parser(
        parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        parser.add_argument(
            '--model_name_or_pth', 
            default=None, 
            help="Path of pipelien to execute. Curretly we only support local directory.", 
        ) 
        parser.add_argument(
            '--trust_remote_code', 
            action='store_true', 
            help="Parameter for huggingface.", 
        )
        parser.add_argument(
            '--seed', 
            default=10086,
            type=int,
            help="Global seed.", 
        )
        parser.add_argument(
            '--use_esymred', 
            action='store_true', 
            help="Whether to use esymred features.", 
        )
        parser.add_argument(
            '--use_batch_split', 
            action='store_true', 
            help="Use batch split feature.", 
        )

        # Parallel configs
        parser.add_argument(
            '--worker_use_ray', 
            default=False,
            action='store_true', 
            help="Workers use ray or native codes.", 
        )
        parser.add_argument(
            '--worker_use_mp', 
            default=True,
            action='store_true', 
            help="Workers use multiprocessing.", 
        )
        parser.add_argument(
            '--pipeline_parallel_size', 
            default=1,
            type=int,
            help="Pipeline parallel size of the whole system.", 
        )
        parser.add_argument(
            '--tensor_parallel_size', 
            default=1,
            type=int,
            help="Tensor parallel size of the whole system.", 
        )
        parser.add_argument(
            '--data_parallel_size', 
            default=1,
            type=int,
            help="Data parallel size of the whole system.", 
        )
        parser.add_argument(
            '--num_cpus_cpu_worker', 
            default=1,
            type=int,
            help="Number of cpus for each extra workers. Extra workers are those "
                 "CPU workers running other tasks such as overlapped prepare stage.", 
        )
        parser.add_argument(
            '--num_cpus_gpu_worker', 
            default=1,
            type=int,
            help="Number of cpus for each gpu workers."
        )
        parser.add_argument(
            '--num_cpu_workers',
            default=1,
            type=int,
            help="Number of cpu workers to use."
        )
        parser.add_argument(
            '--gpus',
            default="[0]",
            type=str,
            help="cuda devices to use, in form like [0-2, 4, 7-8]."
        )

        # Scheduler configs
        parser.add_argument(
            '--max_batchsize', 
            default=16,
            type=int,
            help="Maximum batch size to be scheduled in each round.", 
        )
        parser.add_argument(
            '--use_mixed_precision', 
            action='store_true',
            help="Use mixed precision scheduling. When esymred feature is turned on, "
                 "this will be automatially turned on.", 
        )
        parser.add_argument(
            '--policy', 
            default="fcfs_mixed",
            help="Name of the pocily to use for scheduling.", 
        )

        parser.add_argument(
            '--overlap_prepare', 
            action='store_true',
            help="Whether to overlap prepare stage.", 
        )

        # Engine configs
        parser.add_argument(
            '--disable_log_status', 
            default=False,
            action='store_true',
            help="Disable system's logging.", 
        )

        # kwargs
        parser.add_argument(
            '--torch_dtype', 
            default='float16',
            choices=[torch.float16, torch.float32],
            type=get_torch_dtype_from_string,
            help="Use non blocking paradigm for engine execution.", 
        )
        parser.add_argument(
            '--dispatcher_policy', 
            default="greedy",
            help="Name of the policy for dispatcher.", 
        )

        return parser
    

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # get all the attributes into the form of list
        return cls(**(vars(args)))

    
    def get_pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(
            pipeline=self.model_name_or_pth, 
            trust_remote_code=self.trust_remote_code,
            seed=self.seed, 
            use_esymred=self.use_esymred,
            use_batch_split=self.use_batch_split,
            kwargs=self.kwargs
        )
    
    def get_parallel_config(self) -> ParallelConfig:
        return ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            data_parallel_size=self.data_parallel_size,
            num_cpus_cpu_worker=self.num_cpus_cpu_worker,
            num_cpus_gpu_worker=self.num_cpus_gpu_worker,
            num_cpu_workers=self.num_cpu_workers,
            worker_use_ray=self.worker_use_ray,
            worker_use_mp=self.worker_use_mp,
            gpus=self.gpus,
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
            dispatcher_policy=self.dispatcher_policy,
        )
        
    
    def create_engine_configs(
        self,
    ) -> Tuple[PipelineConfig, ParallelConfig, SchedulerConfig, EngineConfig]:
        pipeline_config = self.get_pipeline_config()
        parallel_config = self.get_parallel_config()
        scheduler_config = self.get_scheduler_config()
        engine_config = self.get_engine_config()
        return pipeline_config, parallel_config, scheduler_config, engine_config


class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous engine, inherited from EngineArgs """
    def __init__(self, **kwargs) -> None:
        self.engine_use_ray = kwargs.pop("engine_use_ray", False)
        self.engine_use_mp = kwargs.pop("engine_use_mp", True)
        self.disable_log_requests = kwargs.pop("disable_log_requests", False)
        self.max_log_len = kwargs.pop("max_log_len", None)
        super().__init__(**kwargs)
    
    
    def get_engine_config(self) -> EngineConfig:
        return EngineConfig(
            log_status=not self.disable_log_status,
            engine_use_ray=self.engine_use_ray,
            engine_use_mp=self.engine_use_mp,
            log_requests=not self.disable_log_requests,
        )
    
    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # add args from base engine
        parser = EngineArgs.add_args_to_parser(parser)
        
        parser.add_argument(
            '--engine_use_ray',
            action='store_true',
            help='Use Ray to start the Execution engine in a separate process '
                'as the server process.'
        )
        parser.add_argument(
            '--engine_use_mp',
            action='store_true',
            help='Use multiprocessing to start the execution engine in a separate process'
                'as the server process.'
        )
        parser.add_argument(
            '--disable_log_requests',
            action='store_true',
            help='disable logging feature of each requests'
        )
        
        return parser
