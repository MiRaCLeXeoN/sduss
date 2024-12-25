import asyncio
import queue

import torch.multiprocessing as mp

from typing import TYPE_CHECKING, Callable, List, Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sduss.engine.engine import Engine
from sduss.entrypoints.wrappers import ReqOutput

from .wrappers import _EngineOutput
from .utils import Task, EngineMainLoop

class _MpAsyncEngine:
    def __init__(
        self,
        *args,
        name: str = "MpAsyncengine",
        **kwargs,
    ) -> None:
        self.name = name

        self.task_queue: 'mp.Queue[Task]' = mp.Queue()
        # Output queue is used for holding request outputs only
        self.output_queue: 'mp.Queue[List[_EngineOutput]]' = mp.Queue()
        # Method res queue is used for any other method's results, 
        # ! results must be in calling order
        self.method_res_queue:'mp.Queue[List[_EngineOutput]]' = mp.Queue()

        # Set afterwards    
        self.engine = mp.Process(
            target=EngineMainLoop,
            name=self.name,
            kwargs={
                "task_queue" : self.task_queue,
                "method_res_queue": self.method_res_queue,
                "output_queue" : self.output_queue,
                "worker_init_fn" : partial(Engine, *args, **kwargs)
            }
        )
        self.engine.start()

        self.thread_executor = ThreadPoolExecutor(1)

    
    def _add_task(self, method_name, need_res: bool, method_args = [], method_kwargs = {}) -> int:
        task = Task(method_name, need_res, *method_args, **method_kwargs)
        self.task_queue.put_nowait(task)
        return task

    
    def _wait_task(self, task) -> Any:
        """Wait until the specified task is finished, and return its result.
        
        Warn:
            This method assumes that the methods will be completed in the order of
            how they are launched. Otherwise, the results will be in a mess.
        """
        # Wait for task to complete
        return self.method_res_queue.get()

        
    def get_output_nowait(self) -> 'List[ReqOutput]':
        try:
            outputs = self.output_queue.get_nowait()
        except queue.Empty:
            outputs = None
        return outputs
    

    def execute_method_sync(
        self, 
        method: str,
        need_res: bool,
        *method_args, 
        **method_kwargs,
    ):
        task = self._add_task(method, need_res, method_args, method_kwargs)
        if need_res:
            return self._wait_task(task)
        else:
            return None
            
    
    async def execute_method_async(
        self, 
        method: str,
        need_res: bool,
        *method_args, 
        **method_kwargs,
    ):
        """Execute the method and wait for its result to be returned."""
        task = self._add_task(method, method_args, method_kwargs)
        return await asyncio.get_running_loop().run_in_executor(
                self.thread_executor, partial(self._wait_task, task)
            )