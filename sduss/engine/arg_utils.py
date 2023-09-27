import argparse
import dataclasses

@dataclasses.dataclass
class EngineArgs:
    """Arguments for the base class Engine
    """
    # ! incomplete
    
    @staticmethod
    def add_args_to_parser(
        parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        # ! incomplete
        
        # parallel arguments
        parser.add_argument(
            '--worker-use-ray',
            action='store_true',
            help='use Ray for distributed serving. Ray will be automatically used when '
                'more than 1 GPU is used'
        )
        
        return parser
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # get all the attributes into the form of list
        attr_names = [attr.name for attr in dataclasses.fields(cls)]
        # generate an instance
        return cls(**{attr_name: getattr(args, attr_name) for attr_name in attr_names})



@dataclasses.dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous engine, inherited from EngineArgs
    """
    # ! incomplete
    engine_use_ray: bool = False
    
    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # add args from base engine
        super().add_args_to_parser(parser)
        
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
        # ! incomplete
        
        return parser