import torch
from torch import distributed as dist
from torch import nn
from models.base_model import BaseModule
import time

class PatchGroupNorm(BaseModule):
    def __init__(self, module: nn.GroupNorm):
        assert isinstance(module, nn.GroupNorm)
        super(PatchGroupNorm, self).__init__(module)
        self.groupnorm_time = 0

   
    def forward(self, input: torch.Tensor, is_sliced: bool = False, image_offset: list=[]) -> torch.Tensor:
        assert input.ndim == 4
        if not is_sliced:
            return self.module(input)
        b, c, h, w = input.shape
        group_size = c // self.module.num_groups
        # reshape mean mean stack
        input = input.view([b, self.module.num_groups, group_size, h, w])
        input_mean = input.mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        input2_mean = (input**2).mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        if is_sliced:
            start = time.time()
            for index in range(len(image_offset) - 1):
                new_input_mean = [input_mean[x] for x in range(image_offset[index], image_offset[index + 1])]
                new_input2_mean = [input2_mean[x] for x in range(image_offset[index], image_offset[index + 1])]
                mean = sum(new_input_mean) / (image_offset[index + 1] - image_offset[index])
                mean2 = sum(new_input2_mean) / (image_offset[index + 1] - image_offset[index])
                for x in range(image_offset[index], image_offset[index + 1]):
                    input_mean[x] = mean
                    input2_mean[x] = mean2
            end = time.time()
            self.groupnorm_time += (end - start)
        var = input2_mean - input_mean**2
        num_elements = group_size * h * w
        var = var * (num_elements / (num_elements - 1))
        std = (var + self.module.eps).sqrt()
        output = (input - input_mean) / std
        output = output.view([b, c, h, w])
        if self.module.affine:
            output = output * self.module.weight.view([1, -1, 1, 1])
            output = output + self.module.bias.view([1, -1, 1, 1])
        # self.counter += 1
        return output