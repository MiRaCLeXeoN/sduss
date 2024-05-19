import torch.multiprocessing as multiprocessing

from typing import TYPE_CHECKING

from sduss.utils import Task, MainLoop

if TYPE_CHECKING:
    from sduss.worker import WorkerOutput

class MpExecutor:
    def __init__(
        self,
        name: str,
        is_prepare_worker: bool,
    ) -> None:
        self.name = name
        self.is_prepare_worker = is_prepare_worker

        self.task_queue: multiprocessing.Queue[Task] = multiprocessing.Queue(100)
        self.output_queue: 'multiprocessing.Queue[WorkerOutput]' = multiprocessing.Queue(100)

        # Set afterwards    
        self.worker = None
    
    
    def init_worker(self, worker_init_fn) -> None:
        self.process = multiprocessing.Process(
            target=MainLoop,
            name=self.name,
            kwargs={
                "task_queue" : self.task_queue,
                "output_queue" : self.output_queue,
                "worker_init_fn" : worker_init_fn,
                },
        )
        self.process.start()
    
    
    def execute_method(self, method: str, *method_args, **method_kwargs):
        self.task_queue.put(Task(method_name=method, *method_args, **method_kwargs))
        return self
    
    
    def get_blocking(self):
        return self.output_queue.get()
    
    
    def data_is_available(self):
        return not self.output_queue.empty()
    
    
    def get_result(self):
        return self.output_queue.get()
    
    
    def end_worker(self):
        self.task_queue.put(Task(method_name="shutdown"))
        self.process.join()