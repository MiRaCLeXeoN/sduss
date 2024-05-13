import torch

class AsyncEngineDeadError(RuntimeError):
    pass


def get_torch_dtype_from_string(dtype_name) -> torch.dtype:
    return getattr(torch, dtype_name)