import uuid
import traceback

from typing import TYPE_CHECKING
from sduss.logger import init_logger

logger = init_logger(__name__)

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


class TaskOutput:

    def __init__(
        self, 
        task_id = None, 
        output = None,
        success = None,
        exception = None,
    ):
        self.id = task_id
        self.output = output
        self.success = success
        self.exception = exception


class RunnerMainLoop:
    """
    If executed method return None, mail loop won't add it
    to the output queue. So please make sure method's return value.

    Warn:
        Main loop works in a "pull" fashion, all results should be 
        explicitly pulled from the partent process! It implies:
            1. Methods and results are produced in exactly the same order!
            2. If no tasks coming, engine will not run at all!
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

        # We must run the main loop in asyncio
        self._main_loop()

    
    def _main_loop(self):
        while True:
            task: Task = self.task_queue.get()
            method_name = task.method

            # Execute method
            try:
                handler = getattr(self.engine, method_name)
                output = handler(*task.args, **task.kwargs)
                engine_output = TaskOutput(task.id, output, True, None)
            except Exception as e:
                engine_output = TaskOutput(task.id, None, False, e)
                logger.error(traceback.format_exc())
                raise e

            if task.need_res or engine_output.exception is not None:
                self.output_queue.put(engine_output)
            
            # If to exit
            if method_name == "shutdown":
                break