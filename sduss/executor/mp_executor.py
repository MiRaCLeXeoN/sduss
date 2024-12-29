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
        self.task_pool: 'Dict[int, Tuple[Task, asyncio.Event]]' = {}

        # Used by background processing loop thread
        self.background_loop = threading.Thread(target=self._process_task_output, name=f"Background taskoutput loop {self.rank}")
        self.wait_task_smp = threading.Semaphore(value=0)
        self.shutdown = False
        self.task_exception = None

    
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
    
    
    def _process_task_output(self):
        while True:
            self.wait_task_smp.acquire(blocking=True)
            if self.shutdown:
                break

            task_output = self.task_res_queue.get()
            # If method failed
            if task_output.exception is not None:
                # ! Since it's run in another thread, main thread cannot automatically detect it
                # ! So we must store it, instead of raising it.
                # We store it here, even if the corresponding task is not awaited,
                # it will be detected by other following tasks.
                self.task_exception = task_output.exception

            task_id = task_output.id
            target_task, event = self.task_pool[task_id]
            # Whichever the task is, we put result of it into the corresponding task
            target_task.output = task_output.output
            del self.task_pool[task_id]
            event.set()
    
    
    def _end_background_loop(self):
        self.shutdown = True
        self.wait_task_smp.release(n=1)
        self.background_loop.join()
    

    def _add_task(self, method_name, need_res: bool, method_args = [], method_kwargs = {}) -> 'Task':
        task = Task(method_name, need_res, *method_args, **method_kwargs)
        self.task_queue.put(task)
        # We only track a task if it need results
        # Exceptions cause by this task will be detected by other coroutines, no need to worry
        if task.need_res:
            event = asyncio.Event()
            self.task_pool[task.id] = (task, event)
            self.wait_task_smp.release(n=1)
        else:
            event = None
        return task, event


    async def _wait_task_res_async(self, task: Task, event) -> Any:
        event.wait()
        if self.task_exception is not None:
            raise self.task_exception
        return task.output


    def execute_method_async(
        self,
        method: str,
        need_res: bool,
        *method_args, 
        **method_kwargs,
    ):
        """Execute method and return the handler for result retrieval.

        Args:
            method (str): method name
            need_res (bool): If set false, no handlers will be returned
        """
        task, event = self._add_task(method, need_res, method_args, method_kwargs)
        # Then we explicitly wait until the result is returned, this will block the whole routine!
        if need_res:
            return asyncio.get_event_loop().create_task(self._wait_task_res_async(task, event))
        else:
            return None
    
    
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

        self._end_background_loop()

        logger.info(f"Worker {self.name} shutdown complete.")


def mp_init_method():
    # Initialize cluster locally
    port = get_open_port()
    distributed_init_method = f"tcp://localhost:{port}"
    return distributed_init_method