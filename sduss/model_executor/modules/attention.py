import math
import time

import xformers
import xformers.ops
import torch
from diffusers.models.attention_processor import Attention
from torch import nn
from torch.nn import functional as F
# from distrifuser.modules.base_module import BaseModule
# from distrifuser.utils import DistriConfig
from .resnet import SplitLinear
from .base_module import BaseModule

class PatchAttention(BaseModule):
    def __init__(self, module: Attention):
        super(PatchAttention, self).__init__(module)
        to_k = module.to_k
        to_v = module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape

        in_size, out_size = to_k.in_features, to_k.out_features
        to_kv = SplitLinear(
            in_size,
            out_size * 2,
            bias=to_k.bias is not None,
            device=to_k.weight.device,
            dtype=to_k.weight.dtype,
        )
        to_kv.weight.data[:out_size].copy_(to_k.weight.data)
        to_kv.weight.data[out_size:].copy_(to_v.weight.data)

        if to_k.bias is not None:
            assert to_v.bias is not None
            to_kv.bias.data[:out_size].copy_(to_k.bias.data)
            to_kv.bias.data[out_size:].copy_(to_v.bias.data)

        self.to_kv = to_kv
        self.attention_time = 0

class PatchCrossAttention(PatchAttention):
    def __init__(self, module: Attention):
        super(PatchCrossAttention, self).__init__(module)
        self.kv_cache = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ):
        
        batch_size, sequence_length, _ = hidden_states.shape
        residual = hidden_states
        query = self.module.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        kv = self.to_kv(encoder_hidden_states)
        # value = attn.to_v(encoder_hidden_states)
        key, value = torch.split(kv, kv.shape[-1] // 2, dim=-1)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.module.heads

        query = self.module.head_to_batch_dim(query).contiguous()
        key = self.module.head_to_batch_dim(key).contiguous()
        value = self.module.head_to_batch_dim(value).contiguous()
        start = time.time()
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value,
        )
        # torch.cuda.synchronize()
        end = time.time()
        self.attention_time += (end - start)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.module.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.module.to_out[0](hidden_states)
        # dropout
        hidden_states = self.module.to_out[1](hidden_states)
        if self.module.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / self.module.rescale_output_factor
        
        return hidden_states

class PatchSelfAttention(PatchAttention):
    def __init__(self, module: Attention):
        super(PatchSelfAttention, self).__init__(module)
        self.self_attention_time = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        scale: float = 1.0,
        is_sliced: bool = False,
        image_offset: list = [],
        resolution_offset: list = [],
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # b, sl, c = hidden_states.shape
        
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
        residual = hidden_states
        states = list()
        kv = self.to_kv(encoder_hidden_states)
        query = self.module.to_q(hidden_states)
        bs = 0
        if is_sliced:
            for resolution_index in range(len(resolution_offset) - 1):
                target_kv = list()
                start = time.time()
                for index in range(resolution_offset[resolution_index], resolution_offset[resolution_index + 1]):
                    patches_per_image = image_offset[index + 1] - image_offset[index]
                    current_kv = torch.cat([kv[x] for x in range(image_offset[index], image_offset[index + 1])], dim=0).unsqueeze(0)
                    for _ in range(image_offset[index], image_offset[index + 1]):
                        target_kv.append(current_kv)
                kv_per_resolution = torch.cat(target_kv, dim=0)
                end = time.time()
                self.self_attention_time += (end - start)
                key, value = torch.split(kv_per_resolution, kv_per_resolution.shape[-1] // 2, dim=-1)
                inner_dim = key.shape[-1]
                query_per_resolution = self.module.head_to_batch_dim(query[bs:bs+key.shape[0]]).contiguous()
                bs = bs+key.shape[0]
                key = self.module.head_to_batch_dim(key).contiguous()
                value = self.module.head_to_batch_dim(value).contiguous()
                states.append(xformers.ops.memory_efficient_attention(
                    query_per_resolution, key, value,
                ))
            hidden_states = torch.cat(states, dim=0)

        else:
            key, value = torch.split(kv, kv.shape[-1] // 2, dim=-1)
            # print(key.shape)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.module.heads

            query = self.module.head_to_batch_dim(query).contiguous()
            key = self.module.head_to_batch_dim(key).contiguous()
            value = self.module.head_to_batch_dim(value).contiguous()
            start = time.time()
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value,
            )
            # torch.cuda.synchronize()
            end = time.time()
            self.attention_time += (end - start)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.module.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.module.to_out[0](hidden_states)
        # dropout
        hidden_states = self.module.to_out[1](hidden_states)
        if self.module.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / self.module.rescale_output_factor
        
        return hidden_states
