import torch.multiprocessing as mp

from typing import Any, Dict, List, Union, Tuple, Type, TYPE_CHECKING, Optional
from functools import partial

from sduss.logger import init_logger

from .utils import Task, TaskOutput, RunnerMainLoop
from ._model_runner import _ModelRunner

logger = init_logger(__name__)

class ModelRunner:
    def __init__(
        self,
        *args,
        name: str = "ModelRunner",
        **kwargs,
    ) -> None:
        self.name = name

        self.task_queue: 'mp.Queue[Task]' = mp.Queue(10)
        # Output queue is used for holding request outputs only
        self.output_queue: 'mp.Queue[TaskOutput]' = mp.Queue(5)

        # Set afterwards    
        self.runner = mp.Process(
            target=RunnerMainLoop,
            name=self.name,
            kwargs={
                "task_queue" : self.task_queue,
                "output_queue" : self.output_queue,
                "worker_init_fn" : partial(_ModelRunner, *args, **kwargs)
            }
        )
        self.runner.start()


    def _add_task(self, method_name, need_res: bool, method_args = [], method_kwargs = {}) -> 'Task':
        task = Task(method_name, need_res, *method_args, **method_kwargs)
        self.task_queue.put_nowait(task)
        return task

    
    def get_result(self, task: Task) -> Any:
        # It's safe to discard all other outputs like this
        # Because we ensure that the worker won't interleave sync and async calls
        # No risk of confusing outputs
        while True:
            # Discard any other outputs until find the target one
            task_output = self.output_queue.get()
            if task_output.exception is not None:
                raise task_output.exception
            if task_output.id == task.id:
                return task_output.output


    def execute_method_sync(
        self, 
        method: str,
        *method_args, 
        **method_kwargs,
    ) -> Any:
        """Execute the method and return the handler.

        Args:
            method (str): method name
            need_res (bool): if true, this method will block until the result is returned,
                otherwise it will return immediately after the task is sent to the engine.
        """
        task = self._add_task(method, True, method_args, method_kwargs)
        res =  self.get_result(task)
        return res

        
    def execute_method_async(
        self, 
        method: str,
        need_res: bool,
        *method_args, 
        **method_kwargs,
    ) -> 'Optional[ModelRunner]':
        """Execute the method and return the handler.

        Args:
            method (str): method name
            need_res (bool): if true, this method will block until the result is returned,
                otherwise it will return immediately after the task is sent to the engine.
        """
        task = self._add_task(method, need_res, method_args, method_kwargs)
        if need_res:
            return task
        else:
            return None
        
    
    def shutdown(self):
        self.execute_method_sync("shutdown")
        self.runner.join()
        logger.info(f"{self.name} shutdown complete")