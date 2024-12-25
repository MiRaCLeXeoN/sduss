import uuid

from typing import TYPE_CHECKING

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
        method_res_queue: 'mp.Queue',
        output_queue: 'mp.Queue',
        worker_init_fn,
    ):
        self.task_queue = task_queue
        self.output_queue = output_queue
        self.method_res_queue = method_res_queue

        self.engine = worker_init_fn()

        self._main_loop()
    
    
    def _main_loop(self):
        while True:
            task: Task = self.task_queue.get()
            method_name = task.method

            if method_name == "shutdown":
                break

            handler = getattr(self.engine, method_name)
            output = handler(*task.args, **task.kwargs)
            self.output_queue.put(output)