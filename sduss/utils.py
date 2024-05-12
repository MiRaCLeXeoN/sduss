import enum
import uuid
import socket

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
    return str(uuid.uuid4().int)

def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]