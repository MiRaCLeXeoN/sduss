import asyncio
import torch.multiprocessing as multiprocessing

from typing import TYPE_CHECKING, List

from sduss.utils import get_open_port
from .utils import Task, ExecutorMainLoop

if TYPE_CHECKING:
    from sduss.worker import WorkerOutput

class MpExecutor:
    def __init__(
        self,
        name: str,
        rank: int,
        device: int,
        is_prepare_worker: bool,
    ) -> None:
        self.name = name
        self.rank = rank,
        self.device = device
        self.is_prepare_worker = is_prepare_worker

        self.task_queue: multiprocessing.Queue[Task] = multiprocessing.Queue(500)
        self.output_queue: 'multiprocessing.Queue[WorkerOutput]' = multiprocessing.Queue(500)

        # Set afterwards    
        self.worker = None
    
    
    def init_worker(self, worker_init_fn) -> None:
        self.process = multiprocessing.Process(
            target=ExecutorMainLoop,
            name=self.name,
            kwargs={
                "task_queue" : self.task_queue,
                "output_queue" : self.output_queue,
                "worker_init_fn" : worker_init_fn,
                },
        )
        self.process.start()
    

    def _add_task(self, method_name, need_res: bool, method_args = [], method_kwargs = {}) -> 'Task':
        task = Task(method_name, need_res, *method_args, **method_kwargs)
        self.task_queue.put_nowait(task)
        return task
    
    
    def execute_method(
        self, 
        method: str,
        need_res: bool,
        *method_args, 
        **method_kwargs,
    ):
        """Execute the method and return the handler.

        Args:
            method (str): method name
            need_res (bool): if true, this method will block until the result is returned,
                otherwise it will return immediately after the task is sent to the engine.
        """
        task = self._add_task(method, need_res, method_args, method_kwargs)
        if need_res:
            # Then we explicitly wait until the result is returned
            return asyncio.get_event_loop().run_until_complete(self._wait_task(task))
        else:
            return None


    
    
    def get_blocking(self):
        return self.output_queue.get()
    
    
    def get_result_nowait(self) -> List:
        outputs = []
        while not self.output_queue.empty():
            outputs.append(self.output_queue.get())
        return outputs
    
    
    def end_worker(self):
        self.task_queue.put(Task(method_name="shutdown"))
        self.process.join()


def mp_init_method():
    # Initialize cluster locally
    port = get_open_port()
    distributed_init_method = f"tcp://localhost:{port}"
    return distributed_init_method