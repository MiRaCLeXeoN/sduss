import uuid
import asyncio

from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from .wrappers import EngineOutput

if TYPE_CHECKING:
    import torch.multiprocessing as mp

class Task:
    def __init__(
        self,
        method_name: str,
        need_res: bool,
        *args,
        **kwargs,
    ):
        """When event is not None, the results are required to put back."""
        self.method = method_name
        self.args = args
        self.kwargs = kwargs
        self.id = uuid.uuid4().int
        self.need_res = need_res


class EngineMainLoop:
    """
    If executed method return None, mail loop won't add it
    to the output queue. So please make sure method's return value.
    """
    def __init__(
        self,
        task_queue: 'mp.Queue',
        output_queue: 'mp.Queue',
        worker_init_fn,
    ):
        self.task_queue = task_queue
        self.output_queue = output_queue

        self.engine = worker_init_fn()

        self.thread_pool = ThreadPoolExecutor(max_workers=1)

        # We must run the main loop in asyncio
        asyncio.get_event_loop().run_until_complete(self._main_loop())
    
    def _get_task(self):
        return self.task_queue.get()
    
    
    async def _main_loop(self):
        while True:
            task: Task = await asyncio.get_event_loop().run_in_executor(self.thread_pool, self._get_task)
            method_name = task.method

            # Execute method
            try:
                handler = getattr(self.engine, method_name)
                output = handler(*task.args, **task.kwargs)
                engine_output = EngineOutput(task.id, output, True, None)
            except Exception as e:
                engine_output = EngineOutput(task.id, None, False, e)

            if task.need_res:
                self.output_queue.put(engine_output)
            
            # If to exit
            if method_name == "clear":
                break

            asyncio.sleep(0)