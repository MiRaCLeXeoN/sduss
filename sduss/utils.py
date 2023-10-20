import enum
import psutil


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

def get_cpu_memory() -> int:
    """Returns the total CPU memory in bytes"""
    return psutil.virtual_memory().total