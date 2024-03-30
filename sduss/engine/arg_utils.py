import argparse

from dataclasses import dataclass, fields
from typing import Optional, Tuple

from sduss.config import PipelineConfig, ParallelConfig, CacheConfig, SchedulerConfig
class EngineArgs:
    """Arguments for the base class Engine
    """
    def __init__(self, model_name_or_pth: str, **kwargs) -> None:
        # Model configs
        self.model_name_or_pth = model_name_or_pth
        self.trust_remote_code = kwargs.pop("trust_remote_code", False)
        self.seed = kwargs.pop("seed", 0)
        # Distributed configs
        self.worker_use_ray = kwargs.pop("worker_use_ray", True)
        self.pipeline_parallel_size = kwargs.pop("pipeline_parallel_size", 1)
        self.tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        self.max_parallel_loading_workers = kwargs.pop("max_parallel_loading_workers", None)
        # Scheduler configs
        self.max_batchsize = kwargs.pop("max_batchsize", 32)
        # Engine configs
        self.disable_log_status = kwargs.pop("disable_log_status", False)
        # kwargs for `from_pretrained`
        self.kwargs = kwargs


    @staticmethod
    def add_args_to_parser(
        parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        # NOTE: If you update any of the arguments below, please also
        # make sure to update docs/source/models/engine_args.rst
        
        # parallel arguments
        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            default='runwayml/stable-diffusion-v1-5',
            help='name or path of the huggingface model to use')
        parser.add_argument('--download-dir',
                            type=str,
                            default=EngineArgs.download_dir,
                            help='directory to download and load the weights, '
                            'default to the default cache dir of huggingface')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='trust remote code from huggingface')
        parser.add_argument(
            '--revision',
            type=str,
            default=None,
            help='the specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=['auto', 'pt', 'safetensors', 'npcache', 'dummy'],
            help='The format of the model weights to load. '
            '"auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available. '
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading. '
            '"dummy" will initialize the weights with random values, '
            'which is mainly for profiling.')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='data type for model weights and activations. '
            'The "auto" option will use FP16 precision '
            'for FP32 and FP16 models, and BF16 precision '
            'for BF16 models.')
        # TODO: Support fine-grained seeds (e.g., seed per request).
        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='random seed')
        
        # Parallel arguments
        parser.add_argument('--worker-use-ray',
                            action='store_true',
                            help='use Ray for distributed serving, will be '
                            'automatically set when using more than 1 GPU')
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='number of pipeline stages')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        parser.add_argument(
            '--max-parallel-loading-workers',
            type=int,
            help='load model sequentially in multiple batches, '
            'to avoid RAM OOM when using tensor '
            'parallel and large models')

        # Scheduler configs
        parser.add_argument('--max-batchsize',
                            type=int,
                            default=EngineArgs.max_batchsize,
                            help='maximum number of sequences per iteration')

        parser.add_argument('--disable-log-status',
                            action='store_true',
                            help="disable engine's periodical status log "
                            "for better performance.")
        return parser
    

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # get all the attributes into the form of list
        attr_names = [attr.name for attr in fields(cls)]
        # generate an instance
        return cls(**{attr_name: getattr(args, attr_name) for attr_name in attr_names})
    
    def create_engine_configs(
        self,
    ) -> Tuple[PipelineConfig, ParallelConfig, SchedulerConfig]:
        model_config = PipelineConfig(self.model_name_or_pth, 
                                   self.trust_remote_code,
                                   self.seed, self.kwargs)
        parallel_config = ParallelConfig(self.pipeline_parallel_size,
                                         self.tensor_parallel_size,
                                         self.worker_use_ray,
                                         self.max_parallel_loading_workers)
        scheduler_config = SchedulerConfig(self.max_batchsize)
        return model_config, parallel_config, scheduler_config


class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous engine, inherited from EngineArgs """
    def __init__(self, model_name_or_pth: str, **kwargs) -> None:
        self.engine_use_ray = kwargs.pop("engine_use_ray", False)
        self.disable_log_requests = kwargs.pop("disable_log_requests", False)
        self.max_log_len = kwargs.pop("max_log_len", None)
        super().__init__(model_name_or_pth, **kwargs)
    
    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # add args from base engine
        parser = EngineArgs.add_cli_args(parser)
        
        parser.add_argument(
            '--engine-use-ray',
            action='store_true',
            help='use Ray to start the Execution engine in a separate process '
                'as the server process'
        )
        parser.add_argument(
            '--disable-log-requests',
            action='store_true',
            help='disable logging requests'
        )
        parser.add_argument('--max-log-len',
                            type=int,
                            default=None,
                            help='max number of prompt characters or prompt '
                            'ID numbers being printed in log. '
                            'Default: unlimited.')
        
        return parser