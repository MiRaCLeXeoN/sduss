import subprocess
import multiprocessing
import time
import ctypes

from typing import Tuple

import torch

class AsyncEngineDeadError(RuntimeError):
    pass


class Metrics:
    def __init__(self, finish_moitor, value, output_queue):
        self.sm_util = list()
        self.finish_moitor = finish_moitor
        self.return_value = value
        self.output_queue = output_queue

    def reset(self):
        self.sm_util = []

    def get_avg_util(self):
        length = len(self.sm_util)
        total = 0
        for util in self.sm_util:
            total += util
        return total / length


class SmUtilMonitor:
    def __init__(
        self,
        to_file_name:str,
        interval:float = 1,
    ) -> None:
        self.file_name = to_file_name
    
        self.finish_moitor = multiprocessing.Value(ctypes.c_int, 0)
        self.value = multiprocessing.Value(ctypes.c_int, 0)
        self.output_queue = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()

        self.metrics = Metrics(self.finish_moitor, self.value, self.output_queue)

        self.process = multiprocessing.Process(
            target=SmUtilMonitor._monitor_sm_utilization, 
            kwargs={"metrics": self.metrics, "lock": self.lock, "interval": interval}
        )
    
    
    @staticmethod
    def get_sm_utilization():
        try:
            # Get nvidia-smi output
            result = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
            sm_utilization = [float(util.strip()) for util in result.decode().split("\n") if util]
            return sm_utilization
        except Exception as e:
            print(f"Error fetching SM utilization: {e}")
            return None
    

    @staticmethod
    def _monitor_sm_utilization(
        metrics : Metrics, 
        lock,
        interval : float = 1,
    ) -> None:
        while True:
            lock.acquire()
            if metrics.finish_moitor.value:
                lock.release()
                break
            if metrics.return_value.value:
                metrics.output_queue.put(metrics.get_avg_util())
                metrics.reset()
                metrics.return_value.value = 0
            sm_utilization = SmUtilMonitor.get_sm_utilization()
            
            if sm_utilization is not None:
                for i, util in enumerate(sm_utilization):
                    # print(f"SM {i} Utilization: {util}%")
                    metrics.sm_util.append(util)
            lock.release()
            time.sleep(interval)
    

    def start_monitor(self):
        self.process.start()
    
    
    def checkpoint(self):
        with self.lock:
            self.value.value = 1
    
    
    def end_monitor(self):
        self.lock.acquire()
        self.finish_moitor.value = 1
        self.lock.release()
        self.process.join()
        

def get_torch_dtype_from_string(dtype_name) -> torch.dtype:
    return getattr(torch, dtype_name)

