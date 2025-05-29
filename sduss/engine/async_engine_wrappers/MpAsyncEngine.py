import asyncio
import queue

import torch.multiprocessing as mp

from typing import TYPE_CHECKING, Callable, List, Any, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sduss.engine.engine import Engine
from sduss.entrypoints.wrappers import ReqOutput
from sduss.logger import init_logger

from .utils import Task, TaskOutput, EngineMainLoop

logger = init_logger(__name__)

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
        self.output_queue: 'mp.Queue[TaskOutput]' = mp.Queue()

        # Set afterwards    
        self.engine = mp.Process(
            target=EngineMainLoop,
            name=self.name,
            kwargs={
                "task_queue" : self.task_queue,
                "output_queue" : self.output_queue,
                "worker_init_fn" : partial(Engine, *args, **kwargs)
            }
        )
        self.engine.start()

        self.thread_executor = ThreadPoolExecutor(1)

    
    def _add_task(self, method_name, need_res: bool, method_args = [], method_kwargs = {}) -> 'Task':
        task = Task(method_name, need_res, *method_args, **method_kwargs)
        self.task_queue.put_nowait(task)
        return task

    
    def _wait_task_output(self, task: Task) -> Any:
        """Wait until the task is finished, and return its result.
        
        Warn:
            This method assumes that the methods will be completed in the order of
            how they are launched. Otherwise, the results will be in a mess.
        """
        # Wait for task to complete
        while True:
            task_output = self.output_queue.get()
            if task_output.exception is not None:
                raise task_output.exception
            if task_output.id == task.id:
                # Discard all other results
                return task_output.output
    

    def execute_method_sync(
        self, 
        method: str,
        *method_args, 
        **method_kwargs,
    ) -> 'Optional[Any]':
        """Execute method, blocking the whole thread until the result is returned.

        Args:
            method (str): method name
            need_res (bool): need result or not

        """
        task = self._add_task(method, True, method_args, method_kwargs)
        # Then we explicitly wait until the result is returned
        # No matter we need result or not
        return self._wait_task_output(task)

        
    async def execute_method_async(
        self, 
        method: str,
        need_res: bool,
        *method_args, 
        **method_kwargs,
    ) -> 'Optional[Any]':
        """Execute the method and return the handler.

        Args:
            method (str): method name
            need_res (bool): if true, this method will block until the result is returned,
                otherwise it will return immediately after the task is sent to the engine.
        """
        task = self._add_task(method, need_res, method_args, method_kwargs)
        if need_res:
            # Then we explicitly wait until the result is returned
            return await asyncio.get_event_loop().run_in_executor(self.thread_executor, partial(self._wait_task_output, task))
        else:
            return None