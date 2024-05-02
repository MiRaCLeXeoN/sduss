import math
import time

from typing import Optional

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
        encoder_hidden_states: torch.FloatTensor or None = None,
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
        self.streams = []
        for _ in range(3):
            self.streams.append(torch.cuda.Stream())
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor or None = None,
        scale: float = 1.0,
        is_sliced: bool = False,
        latent_offset: list = None,
        resolution_offset: list = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        shape = hidden_states.shape
        if len(shape) == 3:
            b, sl, q_c = hidden_states.shape
        else:
            b, h, w, q_c = hidden_states.shape[-1]
        
        residual = hidden_states
        bs = 0
        encoder_hidden_states = hidden_states
        kv = self.to_kv(encoder_hidden_states)
        c = kv.shape[-1]
        query = self.module.to_q(hidden_states)
        if is_sliced:
            states = [None] * (len(resolution_offset) - 1)
            for resolution_index in range(len(resolution_offset) - 1):
                batch_size = latent_offset[resolution_offset[resolution_index + 1]] - latent_offset[resolution_offset[resolution_index]]
                base = 0
                latent_size = resolution_offset[resolution_index + 1] - resolution_offset[resolution_index]
                patch_per_latent = batch_size // latent_size
                # cur_states = hidden_states[latent_offset[resolution_offset[resolution_index]] : latent_offset[resolution_offset[resolution_index + 1]]].view(latent_size, -1, q_c)
                # encoder_hidden_states = cur_states
                # kv = self.to_kv(encoder_hidden_states)
                c = kv.shape[-1]
                # query = self.module.to_q(cur_states)
                ker_per_resolution = kv[latent_offset[resolution_offset[resolution_index]] : latent_offset[resolution_offset[resolution_index + 1]]].view(latent_size, -1, c)
                query_per_resolution = query[latent_offset[resolution_offset[resolution_index]] : latent_offset[resolution_offset[resolution_index + 1]]].view(latent_size, -1, q_c)
                key, value = torch.split(ker_per_resolution, ker_per_resolution.shape[-1] // 2, dim=-1)
                inner_dim = key.shape[-1]
                query_per_resolution = self.module.head_to_batch_dim(query_per_resolution).contiguous()
                bs = bs+key.shape[0]
                key = self.module.head_to_batch_dim(key).contiguous()
                value = self.module.head_to_batch_dim(value).contiguous()
                result = xformers.ops.memory_efficient_attention(query_per_resolution, key, value,)
                result = result.to(query.dtype)
                result = self.module.batch_to_head_dim(result)
                # result = self.module.to_out[0](result)
                states[resolution_index] = result.view(batch_size, -1, q_c)
            hidden_states = torch.cat(states, dim=0)

        else:
            # encoder_hidden_states = hidden_states
            # kv = self.to_kv(encoder_hidden_states)
            # query = self.module.to_q(hidden_states)
            key, value = torch.split(kv, kv.shape[-1] // 2, dim=-1)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.module.heads

            query = self.module.head_to_batch_dim(query).contiguous()
            key = self.module.head_to_batch_dim(key).contiguous()
            value = self.module.head_to_batch_dim(value).contiguous()
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value,
            )
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = self.module.batch_to_head_dim(hidden_states)
        
        hidden_states = self.module.to_out[0](hidden_states)
        # dropout
        hidden_states = self.module.to_out[1](hidden_states)
        if self.module.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / self.module.rescale_output_factor
        return hidden_states
