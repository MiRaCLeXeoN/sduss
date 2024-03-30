import os
import time

from typing import Optional, List, Tuple, Union

import torch
from torch import Tensor
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D

from .base_module import BaseModule

class SplitModule():
    def __init__(self):
        self.profile = None
        self.input_dim = None
    
    def _get_profile(self, directory):
        self.profile = {}
        self.input_dim = {}
        for root, dirs, files in os.walk(directory):
            for file in files:
                if "-throughput" in file:
                    input_dim = file.split("/")[-1].rstrip("-throughput.csv")
                    input_dim = "-".join(list(map(str, input_dim.split("-")[1:])))
                    self.input_dim[input_dim] = input_dim.split("-")
                    # if input_dim not in self.profile:
                    
                    with open(os.path.join(directory, file), "r") as f:
                        start = True
                        lines = f.readlines()
                        if input_dim in self.profile and len(lines) <= len(self.profile[input_dim]) + 1:
                            continue
                        self.profile[input_dim] = []
                        for rec in lines:
                            # rec = f.readline()
                            if rec:
                                if start:
                                    start = False
                                    continue
                                if rec:
                                    elements = rec.split(",")
                                    self.profile[input_dim].append([float(elements[-1]) > 1.04, list(map(int, elements[-2].split("|")))])
                                else:
                                    break

class SplitConv(SplitModule, nn.Conv2d):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int = 1,
                padding: Union[str, int] = 0,
                dilation: int = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros',  # TODO: refine this type
                device=None,
                dtype=None):
        SplitModule.__init__(self)
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.dir = "full-conv"
        self.streams = []
        for _ in range(20):
            self.streams.append(torch.cuda.Stream())
        self.total_conv_time = 0
        
    def get_profile(self, directory):
        if self.padding[0] != 0:
            path = f'{directory}/{self.dir}/{self.in_channels}-{self.out_channels}-{self.kernel_size[0]}-{self.stride[0]}-{self.padding[0]}'
        else:
            path = f'{directory}/{self.dir}/{self.in_channels}-{self.out_channels}-{self.kernel_size[0]}-{self.stride[0]}'
        return self._get_profile(path)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # print(f'conv {self.in_channels}|{self.out_channels}|{self.kernel_size[0]}|{self.stride[0]}|{self.padding[0]} {input.shape[0]}|{input.shape[1]}|{input.shape[2]}|{input.shape[3]}')
        start = time.time()
        if self.profile is not None:
            shape = input.shape
            batch_size = shape[0]
            input_dim = "-".join(list(map(str, shape[1:])))
            if input_dim in self.profile and self.profile[input_dim][batch_size - 1][0]:
                
                split_inputs = input.split(self.profile[input_dim][batch_size - 1][1], dim=0)
                length = len(split_inputs)
                res = [None] * length
                for index in range(length):
                    with torch.cuda.stream(self.streams[index]):
                        res[index] = F.conv2d(split_inputs[index], weight, bias, self.stride, self.padding, self.dilation, self.groups)
                for index in range(length):
                    self.streams[index].synchronize()
                return torch.cat(res, dim=0)
        c = F.conv2d(input, weight, bias, self.stride,
           self.padding, self.dilation, self.groups)
        # torch.cuda.synchronize()
        end = time.time()
        self.total_conv_time += (end - start)
        return c
    
    def patched_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], padding: bool):
        if self.profile is not None:
            shape = input.shape
            batch_size = shape[0]
            input_dim = "-".join(list(map(str, shape[1:])))
            if self.profile[input_dim][batch_size - 1][0]:
                
                split_inputs = input.split(self.profile[input_dim][batch_size - 1][1], dim=0)
                length = len(split_inputs)
                res = [None] * length
                for index in range(length):
                    with torch.cuda.stream(self.streams[index]):
                        res[index] = F.conv2d(split_inputs[index], weight, bias, stride=self.stride, padding=self.padding if padding else (0,0))
                for index in range(length):
                    self.streams[index].wait()
                return torch.cat(res, dim=0)
        
        return F.conv2d(input, weight, bias, stride=self.stride,
           padding=self.padding if padding else (0,0))

class SplitLinear(SplitModule, nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        SplitModule.__init__(self)
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)
        self.dir = "linear"
        self.streams = []
        for _ in range(20):
            self.streams.append(torch.cuda.Stream())
        self.total_linear_time = 0
        
    
    def get_profile(self, directory):
        path = f'{directory}/{self.dir}/{self.in_features}-{self.out_features}'
        return self._get_profile(path)

    def forward(self, input: Tensor):
        # if input.ndim == 3:
        #     print(f'linear {self.in_features}|{self.out_features} {input.shape[0]}|{input.shape[1]}|{input.shape[2]}')
        # else:
        #     print(f'linear {self.in_features}|{self.out_features} {input.shape[0]}|{input.shape[1]}')
        start = time.time()
        if self.profile is not None:
            shape = input.shape
            batch_size = shape[0]
            input_dim = "-".join(list(map(str, shape[1:])))
            if input_dim in self.profile and self.profile[input_dim][batch_size - 1][0]:
                
                split_inputs = input.split(self.profile[input_dim][batch_size - 1][1], dim=0)
                length = len(split_inputs)
                res = [None] * length
                for index in range(length):
                    with torch.cuda.stream(self.streams[index]):
                        res[index] = F.linear(split_inputs[index], self.weight, self.bias)

                for index in range(length):
                    self.streams[index].synchronize()
                return torch.cat(res, dim=0)
        l = F.linear(input, self.weight, self.bias)
        # torch.cuda.synchronize()
        end = time.time()
        self.total_linear_time += (end - start)
        return l

class SplitGroupnorm(SplitModule, nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None):
        SplitModule.__init__(self)
        nn.GroupNorm.__init__(self, num_groups, num_channels, eps, affine, device, dtype)
        self.dir = "groupnorm"
        self.total_groupnorm_time = 0
        self.streams = []
        for _ in range(20):
            self.streams.append(torch.cuda.Stream())
        
    
    def get_profile(self, directory):
        path = f'{directory}/{self.dir}/{self.num_groups}-{self.num_channels}'
        return self._get_profile(path)

    def forward(self, input: Tensor):
        # print(f'linear {self.num_groups}|{self.num_channels} {input.shape[0]}|{input.shape[1]}|{input.shape[2]}|{input.shape[3]}')
        start = time.time()
        if self.profile is not None:
            shape = input.shape
            batch_size = shape[0]
            input_dim = "-".join(list(map(str, shape[1:])))
            if input_dim in self.profile and self.profile[input_dim][batch_size - 1][0]:
                
                split_inputs = input.split(self.profile[input_dim][batch_size - 1][1], dim=0)
                length = len(split_inputs)
                res = [None] * length
                for index in range(length):
                    with torch.cuda.stream(self.streams[index]):
                        res[index] = F.group_norm(split_inputs[index], self.num_groups, self.weight, self.bias, self.eps)
                for index in range(length):
                    self.streams[index].synchronize()
                return torch.cat(res, dim=0)
        gn = F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        # torch.cuda.synchronize()
        end = time.time()
        self.total_groupnorm_time += (end - start)
        return gn
    
class SplitBmm(SplitModule, nn.Module):
    def __init__(self):
        SplitModule.__init__(self)
        nn.Module.__init__(self)
        self.dir = "bmm"
    
    def get_profile(self, directory):
        path = f'{directory}/{self.dir}'
        return self._get_profile(path)

    def forward(self, input1: Tensor, input2: Tensor):
        if self.profile is not None:
            shape = input2.shape
            batch_size = shape[0]
            input_dim = "-".join(list(map(str, shape[1:])))
            if input_dim in self.profile and self.profile[input_dim][batch_size - 1][0]:
                res = []
                split_inputs1 = input1.split(self.profile[input_dim][batch_size - 1][1], dim=0)
                split_inputs2 = input2.split(self.profile[input_dim][batch_size - 1][1], dim=0)
                for index in range(len(split_inputs1)):
                    res.append(torch.bmm(split_inputs1[index], split_inputs2[index]))
                return torch.cat(res, dim=0)
        
        return torch.bmm(input1, input2)

class PatchConv(BaseModule):
    def __init__(self, module: nn.Conv2d):
        super(PatchConv, self).__init__(module)
        self.conv_time = 0

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: [B, C, H, W]
        output = self.module(x)
        return output

    # padding_idx: has 4 sub-lists coresponding with four direction, 
    # each of which has the index of batch the origin batch should padding with
    def forward(self, input: torch.Tensor, is_sliced: bool=False, padding_idx: list=[], *args, **kwargs) -> torch.Tensor:
        b, c, h, w = input.shape
        boundary_size = self.module.padding[0]
        conv_list = list()
        if not is_sliced or boundary_size == 0:
            output = self.naive_forward(input)
        else:
            def find_padding_batch(batch_arr, direction):
                # padding left
                if direction == 0:
                    return torch.cat(
                            [
                                torch.cat([input[x : x + 1, :, :1, -boundary_size:] if x != -1 else torch.zeros([1, c, 1, boundary_size], device="cuda", dtype=torch.float16) for x in batch_arr], dim=0),
                                torch.cat([input[x : x + 1, :, :, -boundary_size:] if x != -1 else torch.zeros([1, c, h, boundary_size], device="cuda", dtype=torch.float16) for x in batch_arr], dim=0),
                                torch.cat([input[x : x + 1, :, -1:, -boundary_size:] if x != -1 else torch.zeros([1, c, 1, boundary_size], device="cuda", dtype=torch.float16) for x in batch_arr], dim=0)
                            ],
                            dim=2
                        )
                # padding top
                elif direction == 1:
                    return torch.cat([input[x : x +1 , :, -boundary_size:, :] if x != -1 else torch.zeros([1, c, boundary_size, w], device="cuda", dtype=torch.float16) for x in batch_arr], dim=0)
                # padding right
                elif direction == 2:
                    return torch.cat(
                        [
                            torch.cat([input[x : x + 1, :, :1, :boundary_size] if x != -1 else torch.zeros([1, c, 1, boundary_size], device="cuda", dtype=torch.float16) for x in batch_arr], dim=0),
                            torch.cat([input[x : x + 1, :, :, :boundary_size] if x != -1 else torch.zeros([1, c, h, boundary_size], device="cuda", dtype=torch.float16) for x in batch_arr], dim=0),
                            torch.cat([input[x : x + 1, :, -1:, :boundary_size] if x != -1 else torch.zeros([1, c, 1, boundary_size], device="cuda", dtype=torch.float16) for x in batch_arr], dim=0)
                        ],
                        dim=2
                    )
                # padding bottom
                elif direction == 3:
                    return torch.cat([input[x : x + 1, :, :boundary_size, :] if x != -1 else torch.zeros([1, c, boundary_size, w], device="cuda", dtype=torch.float16) for x in batch_arr], dim=0)
            def create_padded_x():
                return torch.cat(
                    [
                        find_padding_batch(padding_idx[0], 0),
                        torch.cat(
                            [
                                find_padding_batch(padding_idx[1], 1),
                                input,
                                find_padding_batch(padding_idx[3], 3)
                            ],
                            dim=2
                        ),
                        find_padding_batch(padding_idx[2], 2)
                    ],
                    dim=3
                )
            start = time.time()
            padded_x = create_padded_x()
            end = time.time()
            self.conv_time += (end - start)
            output = self.module.patched_forward(padded_x, self.module.weight, self.module.bias, False)


        return output

class PatchUpsample2D(BaseModule):

    def __init__(self, module: Upsample2D):
        super().__init__(module)


    def forward(self, hidden_states, 
                output_size=None, 
                image_offset: list = [],
                padding_idx: list = [],
                is_sliced:bool = False):
        assert hidden_states.shape[1] == self.module.channels

        if self.module.norm is not None:
            hidden_states = self.module.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.module.use_conv_transpose:
            return self.module.conv(hidden_states, is_sliced=is_sliced, padding_idx=padding_idx)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.module.use_conv:
            if self.module.name == "conv":
                hidden_states = self.module.conv(hidden_states, is_sliced=is_sliced, padding_idx=padding_idx)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

class PatchDownsample2D(BaseModule):
    def __init__(self, module: Downsample2D):
        super().__init__(module)
    
    def forward(self, hidden_states,
                image_offset: list = [],
                padding_idx: list = [],
                is_sliced:bool = False):
        assert hidden_states.shape[1] == self.module.channels
        if self.module.norm is not None:
            hidden_states = self.module.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.module.use_conv and self.module.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.module.channels
        hidden_states = self.module.conv(hidden_states, is_sliced=is_sliced, padding_idx=padding_idx)

        return hidden_states

class PatchResnetBlock2D(BaseModule):
    def __init__(
        self,
        module: ResnetBlock2D
    ):
        super().__init__(module)
    
    def forward(self, input_tensor, temb, 
                image_offset: list = [],
                padding_idx: list = [],
                is_sliced:bool = False):
        hidden_states = input_tensor

        hidden_states = self.module.norm1(hidden_states, is_sliced=is_sliced, image_offset=image_offset)
        hidden_states = self.module.nonlinearity(hidden_states)

        if self.module.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.module.upsample(input_tensor, is_sliced=is_sliced, image_offset=image_offset, padding_idx=padding_idx)
            hidden_states = self.module.upsample(hidden_states, is_sliced=is_sliced, image_offset=image_offset, padding_idx=padding_idx)
        elif self.module.downsample is not None:
            input_tensor = self.module.downsample(input_tensor, is_sliced=is_sliced, image_offset=image_offset, padding_idx=padding_idx)
            hidden_states = self.module.downsample(hidden_states, is_sliced=is_sliced, image_offset=image_offset, padding_idx=padding_idx)

        hidden_states = self.module.conv1(hidden_states, is_sliced=is_sliced, padding_idx=padding_idx)

        if temb is not None:
            temb = self.module.time_emb_proj(self.module.nonlinearity(temb))[:, :, None, None]

        if self.module.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb

            hidden_states = self.module.norm2(hidden_states, is_sliced=is_sliced, image_offset=image_offset)
        elif self.module.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.module.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.module.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.module.norm2(hidden_states)

        # if temb is not None and self.module.time_embedding_norm == "scale_shift":
        #     scale, shift = torch.chunk(temb, 2, dim=1)
        #     hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.module.nonlinearity(hidden_states)

        hidden_states = self.module.dropout(hidden_states)
        hidden_states = self.module.conv2(hidden_states, is_sliced=is_sliced, padding_idx=padding_idx)

        if self.module.conv_shortcut is not None:
            input_tensor = self.module.conv_shortcut(input_tensor, is_sliced=is_sliced, padding_idx=padding_idx)

        output_tensor = (input_tensor + hidden_states) / self.module.output_scale_factor

        return output_tensor