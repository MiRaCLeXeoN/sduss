import asyncio
import threading
import torch.multiprocessing as multiprocessing

from typing import TYPE_CHECKING, List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

from sduss.utils import get_open_port
from sduss.logger import init_logger
from .utils import Task, ExecutorMainLoop

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
    ) -> None:
        self.name = name
        self.rank = rank,
        self.device_num = device
        self.is_prepare_worker = is_prepare_worker

        self.task_queue: multiprocessing.Queue[Task] = multiprocessing.Queue(20)
        self.task_res_queue: multiprocessing.Queue[Task] = multiprocessing.Queue(20)
        self.output_queue: 'multiprocessing.Queue[WorkerOutput]' = multiprocessing.Queue(100)

        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.task_pool: Dict[int, Tuple[Task, asyncio.Event]] = {}
        self.task_semaphore = threading.Semaphore(value=0)
        self.shutdown_event = threading.Event()

        self.task_res_loop = asyncio.get_event_loop().run_in_executor(self.thread_pool, self._process_task_res_loop)

    
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
    

    def _process_task_res_loop(self) -> None:
        while True:
            # Wait for task to come
            self.task_semaphore.acquire(blocking=True)
            # Check if to exit
            if self.shutdown_event.is_set():
                break

            task_output = self.task_res_queue.get()
            task_id = task_output.id
            with self.lock:
                task, event = self.task_pool.pop(task_id)
            task.output = task_output.output
            # Awake the corresponding coroutine
            event.set()
        

    def _add_task(self, method_name, method_args = [], method_kwargs = {}) -> 'Task':
        task = Task(method_name, *method_args, **method_kwargs)
        self.task_queue.put(task)
        event = asyncio.Event()

        with self.lock:
            self.task_pool[task.id] = (task, event)

        self.task_semaphore.release(n=1)
        return task, event
    
    
    async def _wait_task(self, task, event) -> Any:
        event.wait()
        return task.output
    
    
    def execute_method_nonblocking(
        self, 
        method: str,
        *method_args, 
        **method_kwargs,
    ):
        """Execute the method and return the future.

        Args:
            method (str): method name
            need_res (bool): if true, this method will block until the result is returned,
                otherwise it will return immediately after the task is sent to the engine.
        """
        task, event = self._add_task(method, method_args, method_kwargs)
        # This requires the engine run in async fashion! 
        # If engine is not, this coroutine will never get chance to run
        return asyncio.get_event_loop().create_task(self._wait_task(task, event))


    def execute_method_blocking(
        self,
        method: str,
        *method_args, 
        **method_kwargs,
    ):
        task, event = self._add_task(method, method_args, method_kwargs)
        # Then we explicitly wait until the result is returned
        return asyncio.get_event_loop().run_until_complete(self._wait_task(task, event))
    
    
    def get_output_nowait(self) -> 'List[WorkerOutput]':
        outputs = []
        while not self.output_queue.empty():
            outputs.append(self.output_queue.get())
        return outputs
    
    
    def end_worker(self):
        # End worker process
        self.task_queue.put(Task(method_name="shutdown"))
        self.process.join()

        # End output loop
        self.task_semaphore.release(n=1)
        self.shutdown_event.set()
        asyncio.get_event_loop().run_until_complete(self.task_res_loop)

        logger.info(f"Worker {self.name} shutdown.")


def mp_init_method():
    # Initialize cluster locally
    port = get_open_port()
    distributed_init_method = f"tcp://localhost:{port}"
    return distributed_init_method