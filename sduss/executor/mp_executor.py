import asyncio
import threading
import torch.multiprocessing as multiprocessing

from typing import TYPE_CHECKING, List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

from sduss.utils import get_open_port
from sduss.logger import init_logger
from .utils import Task, ExecutorMainLoop
from .wrappers import TaskOutput

if TYPE_CHECKING:
    from sduss.worker import WorkerOutput

logger = init_logger(__name__)

class MpExecutor:
    def __init__(
        self,
        name: str,
        rank: int,
        device: int,
        is_prepare_worker: bool,
        thread_pool = None,
    ) -> None:
        self.name = name
        self.rank = rank,
        self.device_num = device
        self.is_prepare_worker = is_prepare_worker

        self.task_queue: multiprocessing.Queue[Task] = multiprocessing.Queue(20)
        self.task_res_queue: multiprocessing.Queue[TaskOutput] = multiprocessing.Queue(20)
        self.output_queue: 'multiprocessing.Queue[WorkerOutput]' = multiprocessing.Queue(100)

        if thread_pool:
            self.thread_pool = thread_pool
        else:
            self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.task_pool: Dict[int, Task] = {}

    
    def init_worker(self, worker_init_fn) -> None:
        self.process = multiprocessing.Process(
            target=ExecutorMainLoop,
            name=self.name,
            kwargs={
                "task_queue" : self.task_queue,
                "task_res_queue" : self.task_res_queue,
                "output_queue" : self.output_queue,
                "worker_init_fn" : worker_init_fn,
                },
        )
        self.process.start()
    

    def _add_task(self, method_name, method_args = [], method_kwargs = {}) -> 'Task':
        task = Task(method_name, *method_args, **method_kwargs)
        self.task_queue.put(task)
        self.task_pool[task.id] = task
        return task


    def _wait_task_res_sync(self, task) -> Any:
        if task.is_finished:
            return task.ouput
        else:
            # Process the res until the target is found
            while not task.is_finished:
                task_output = self.task_res_queue.get()
                # If method failed
                if task_output.exception is not None:
                    raise task_output.exception
                task_id = task_output.id
                # Whichever the task is, we put result of it into the corresponding task
                target_task = self.task_pool[task_id]
                target_task.output = task_output.output
                target_task.is_finished = True
            # Get the result we want
            del self.task_pool[task.id]
            return task.output
    
    
    def execute_method_sync(
        self,
        method: str,
        *method_args, 
        **method_kwargs,
    ):
        """Execute the method and block the whole thread until result is returned.

        Args:
            method (str): method name

        Warn:
            This call will block the whole process!
        """
        task = self._add_task(method, method_args, method_kwargs)
        # Then we explicitly wait until the result is returned, this will block the whole routine!
        return self._wait_task_res_sync(task)
    
    
    async def _wait_task_res_async(self, task) -> Any:
        # * Since all coroutines are run within the same thread, we dont have to
        # * worry about race conditions!
        # * Only one coroutine will modify shared varaibles concurrently

        if task.is_finished:
            return task.ouput
        else:
            # Process the res until the target is found
            while not task.is_finished:
                task_output = await asyncio.get_event_loop().run_in_executor(self.thread_pool, self.task_res_queue.get)
                # If method failed
                if task_output.exception is not None:
                    raise task_output.exception
                task_id = task_output.id
                # Whichever the task is, we put result of it into the corresponding task
                target_task = self.task_pool[task_id]
                target_task.output = task_output.output
                target_task.is_finished = True
            # Get the result we want
            return task.output


    def execute_method_async(
        self,
        method: str,
        *method_args, 
        **method_kwargs,
    ):
        """Execute method and return the handler for result retrieval.

        Args:
            method (str): method name
        """
        task = self._add_task(method, method_args, method_kwargs)
        # Then we explicitly wait until the result is returned, this will block the whole routine!
        return asyncio.get_event_loop().create_task(self._wait_task_res_async(task))
    
    
    def get_output_nowait(self) -> 'List[WorkerOutput]':
        outputs = []
        while not self.output_queue.empty():
            outputs.append(self.output_queue.get())
        return outputs
    
    
    def end_worker(self):
        # End worker process
        logger.info(f"Worker {self.name} shutdown start.")
        self.task_queue.put(Task(method_name="shutdown"))
        self.process.join()

        logger.info(f"Worker {self.name} shutdown complete.")


def mp_init_method():
    # Initialize cluster locally
    port = get_open_port()
    distributed_init_method = f"tcp://localhost:{port}"
    return distributed_init_method