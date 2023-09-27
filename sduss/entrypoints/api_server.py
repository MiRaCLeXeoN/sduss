import argparse

from sduss.engine.arg_utils import AsyncEngineArgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="sduss api server entry",
        description="This python program launches an API server to accept and "
            "process request for model inference",
    )
    parser.add_argument('--host', type=str, default='localhost', help='host address')
    parser.add_argument('--port', type=int, default=6888, help='port')
    parser = AsyncEngineArgs.add_args_to_parser(parser)
    
    args = parser.parse_args()
    
    # instantiate an engine
    engine = 
    
    parser.parse_args()