import uuid
import asyncio
import queue    

from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from .wrappers import TaskOutput

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
        self.output = None


class ExecutorMainLoop:
    """
    If executed method return None, mail loop won't add it
    to the output queue. So please make sure method's return value.
    """
    def __init__(
        self,
        task_queue: 'mp.Queue',
        output_queue: 'mp.Queue',
        task_res_queue: 'mp.Queue',
        worker_init_fn,
    ):
        self.task_queue = task_queue
        self.output_queue = output_queue
        self.task_res_queue = task_res_queue

        self.worker = worker_init_fn()

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.new_reqs_event = asyncio.Event()
        # We must run 2 loops at the same time
        # Because we want 1) step is always proceeding, in a "pushing" not "pulling" fashion
        # 2) even there are no tasks ongoing, it can still respond to any tasks
        self.schedule_loop = asyncio.get_event_loop().create_task(self._schedule_loop())
        asyncio.get_event_loop().run_until_complete(self._main_loop())

    
    async def _schedule_loop(self):
        try:
            have_reqs = False
            while True:
                # To avoid busy waiting, we use event here
                if not have_reqs:
                    await self.new_reqs_event.wait()
                self.new_reqs_event.clear()
                worker_output, have_reqs = self.worker.step()
                if worker_output is not None:
                    self.output_queue.put(TaskOutput(None, worker_output, True, None))
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise

    
    def _process_task(self, task: Task):
        method_name = task.method

        # Execute method
        try:
            handler = getattr(self.worker, method_name)
            output = handler(*task.args, **task.kwargs)
            task_output = TaskOutput(task.id, output, True, None)
        except Exception as e:
            task_output = TaskOutput(task.id, None, False, e)
        
        if method_name == "add_requests":
            self.new_reqs_event.set()
        
        return task_output
    

    async def _main_loop(self):
        while True:
            task: Task = await asyncio.get_event_loop().run_in_executor(self.thread_pool, self.task_queue.get)
            task_output = self._process_task(task)
            # Even if the caller doesn't need result, we still return an output if
            # there is an exception
            if task.need_res or task_output.exception is not None:
                self.task_res_queue.put(task_output)

            # If to exit
            if task.method == "shutdown":
                self.schedule_loop.cancel()
                break

            await asyncio.sleep(0)
            