import time

import torch

from torch import distributed as dist
from torch import nn
from torch.utils.cpp_extension import load

from .base_module import BaseModule

esymred = load(
    name="esymred",
    sources=[
        "models/kernels/norm_silu_concat.cpp",
        "models/kernels/norm_silu_concat.cu",
    ],
    verbose=True,
    extra_ldflags= ["-fopenmp"],
    extra_cflags = ["-fopenmp"],
    extra_cuda_cflags = [" --ptxas-options=-v --extended-lambda"],
    extra_include_paths = ["/home/zzp/miniconda3/envs/sduss/lib/python3.9/site-packages/torch/include"])

class PatchGroupNorm(BaseModule):
    def __init__(self, module: nn.GroupNorm):
        assert isinstance(module, nn.GroupNorm)
        super(PatchGroupNorm, self).__init__(module)
        self.groupnorm_time = 0

   
    def forward(self, input: torch.Tensor, is_sliced: bool = False, latent_offset: torch.Tensor=None, patch_map: torch.Tensor = None, padding_idx: torch.Tensor = None, is_fused: bool = True) -> torch.Tensor:
        assert input.ndim == 4
        if not is_sliced or not is_fused:
            result = self.module(input)
            return result
        else:
            N, C, H, W = input.shape
            result = esymred.groupnorm(input, self.module.weight, self.module.bias, N, C, H, W, int(C / self.module.num_groups), self.module.eps, latent_offset, patch_map, padding_idx)
            return result