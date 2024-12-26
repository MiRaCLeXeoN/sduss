import uuid
import asyncio

from typing import TYPE_CHECKING

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


class ExecutorMainLoop:
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

        self.worker = worker_init_fn()

        self.new_reqs_event = asyncio.Event()
        self._background_loop_unshield = asyncio.get_event_loop().create_task()

        self._main_loop()
    
    
    async def _schedule_loop(self):
        have_reqs = False
        while True:
            if not have_reqs:
                await self.new_reqs_event.wait()
            worker_output = self.worker.step()
            if worker_output is not None:
                self.output_queue.put(worker_output)
            

                
            
    
    
    def _main_loop(self):
        while True:
            task: Task = self.task_queue.get()
            method_name = task.method

            # Execute method
            try:
                handler = getattr(self.engine, method_name)
                output = handler(*task.args, **task.kwargs)
            except Exception as e:
                engine_output = EngineOutput(task.id, None, False, e)
            finally:
                engine_output = EngineOutput(task.id, output, True, None)

            if task.need_res:
                self.output_queue.put(engine_output)
            
            # If to exit
            if method_name == "clear":
                break