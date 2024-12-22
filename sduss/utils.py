import enum
import uuid
import socket
import os
import multiprocessing

from platform import uname

import psutil
import torch

class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()

class Counter:
    
    def __init__(self, start: int = 0) -> None:
        self.counter = start
    
    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i
    
    def reset(self) -> None:
        self.counter = 0

class Task:
    def __init__(
        self,
        method_name: str,
        *args,
        **kwargs,
    ):
        self.method = method_name
        self.args = args
        self.kwargs = kwargs


class MainLoop:
    """
    If executed method return None, mail loop won't add it
    to the output queue. So please make sure method's return value.
    """
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
            self.output_queue.put(output)


def is_hip() -> bool:
    return torch.version.hip is not None

def get_cpu_memory() -> int:
    """Returns the total CPU memory in bytes"""
    return psutil.virtual_memory().total

def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total gpu memory in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory

def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()

def random_uuid() -> int:
    return uuid.uuid4().int

def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_os_env(name: str, check_none: bool = False):
    var = os.getenv(name)
    if check_none:
        assert var is not None, f"Trying to retrieve {name} from environment variables but got None."
    return var