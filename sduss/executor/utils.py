import uuid
import asyncio
import queue    

from typing import TYPE_CHECKING

from .wrappers import TaskOutput

if TYPE_CHECKING:
    import torch.multiprocessing as mp

class Task:
    def __init__(
        self,
        method_name: str,
        *args,
        **kwargs,
    ):
        """When event is not None, the results are required to put back."""
        self.method = method_name
        self.args = args
        self.kwargs = kwargs
        self.id = uuid.uuid4().int


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

        self.new_reqs_event = asyncio.Event()
        asyncio.get_event_loop().run_until_complete(self._main_loop())

        # TODO: Try separate main loop and schedule loop
    
    
    async def _schedule_loop(self):
        have_reqs = False
        while True:
            if not have_reqs:
                await self.new_reqs_event.wait()
            self.new_reqs_event.clear()
            worker_output, have_reqs = self.worker.step()
            if worker_output is not None:
                self.output_queue.put(TaskOutput(None, worker_output, True, None))
            asyncio.sleep(0)

    
    def _process_task(self, task: Task):
        method_name = task.method

        # Execute method
        try:
            handler = getattr(self.worker, method_name)
            output = handler(*task.args, **task.kwargs)
            task_output = TaskOutput(task.id, output, True, None)
        except Exception as e:
            task_output = TaskOutput(task.id, None, False, e)
        
        # if method_name == "add_requests":
        #     self.new_reqs_event.set()
        
        return task_output
    
    
    async def _main_loop(self):
        # FIXME: If we execute this, when will background loop be executed?
        while True:
            # 1. Check if there is task to do
            try:
                task: Task = self.task_queue.get_nowait()
            except queue.Empty:
                task = None
            if task is not None:
                if task.method == "shutdown":
                    break

                task_output = self._process_task(task)
                self.task_res_queue.put(task_output)
                # If to exit
            
            # 2. One schedule step
            worker_output, have_reqs = self.worker.step()
            if worker_output is not None:
                self.output_queue.put(TaskOutput(None, worker_output, True, None))