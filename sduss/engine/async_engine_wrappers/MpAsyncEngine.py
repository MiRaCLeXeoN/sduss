import asyncio

import torch.multiprocessing as multiprocessing

from typing import TYPE_CHECKING, Callable, List
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sduss.utils import Task, MainLoop
from sduss.engine.engine import Engine

if TYPE_CHECKING:
    from sduss.worker import WorkerOutput
    

class _MpAsyncEngine:
    def __init__(
        self,
        *args,
        name: str = "MpAsyncengine",
        **kwargs,
    ) -> None:
        self.name = name

        self.task_queue: multiprocessing.Queue[Task] = multiprocessing.Queue()
        self.output_queue: 'multiprocessing.Queue[List[WorkerOutput]]' = multiprocessing.Queue()

        # Set afterwards    
        self.engine = multiprocessing.Process(
            target=MainLoop,
            name=self.name,
            kwargs={
                "task_queue" : self.task_queue,
                "output_queue" : self.output_queue,
                "worker_init_fn" : partial(Engine, *args, **kwargs)
            }
        )
        self.engine.start()

        self.thread_executor = ThreadPoolExecutor(1)

    
    
    def get_result(self, method_name, method_args = [], method_kwargs = {}):
        self.task_queue.put_nowait(
            Task(method_name, *method_args, **method_kwargs)
        )
        return self.output_queue.get()
    
    
    async def execute_method(
        self, 
        method: str, 
        *method_args, 
        **method_kwargs,
    ):
        return await asyncio.get_running_loop().run_in_executor(
                self.thread_executor, partial(self.get_result, method, method_args, method_kwargs)
            )
        
    