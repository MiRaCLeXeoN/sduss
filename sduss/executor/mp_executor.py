import multiprocessing

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sduss.worker import WorkerOutput

class Task:
    def __init__(
        self,
        method: str,
        *args,
        **kwargs,
    ):
        self.method = method
        self.args = args
        self.kwargs = kwargs
    

class MainLoop:
    def __init__(
        self,
        task_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        worker_init_fn,
    ):
        self.task_queue = task_queue
        self.output_queue = output_queue

        self.worker = worker_init_fn()

        self._main_loop()
    
    
    def _main_loop(self):
        while True:
            task: Task = self.task_queue.get()
            method_name = task.method

            if method_name == "shutdown":
                break

            handler = getattr(self.worker, method_name)
            output = handler(*task.args, **task.kwargs)
            self.output_queue.put_nowait(output)
    

class MpExecutor:
    def __init__(
        self,
        name: str,
        is_prepare_worker: bool,
    ) -> None:
        self.name = name
        self.is_prepare_worker = is_prepare_worker

        self.task_queue: multiprocessing.Queue[Task] = multiprocessing.Queue()
        self.output_queue: 'multiprocessing.Queue[WorkerOutput]' = multiprocessing.Queue()

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
    
    
    def execute_method(self, method: str, *method_args, **method_kwargs):
        self.task_queue.put_nowait(Task(method=method, *method_args, **method_kwargs))
        return self.get_result
    
    
    def get_result(self):
        return self.output_queue.get()
    
    
    def start_worker(self):
        self.process.start()
    
    
    def end_worker(self):
        self.process.join()