import time
import torch
import math

from typing import Optional

import xformers
import xformers.ops
import numpy as np

from torch import nn
from torch.nn import functional as F
from diffusers.models.attention_processor import Attention
from torch.utils.cpp_extension import load
# from distrifuser.modules.base_module import BaseModule
# from distrifuser.utils import DistriConfig

from .resnet import SplitLinear
from .base_module import BaseModule
from .cache_manager import CacheManager


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
        self.cross_attn_output = CacheManager()
        self.cross_attn_time = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        mask: list = None,
        input_indices: list = None,
        *args,
        **kwargs,
    ):
        # torch.cuda.synchronize()
        # start = time.time()
        batch_size, sequence_length, _ = hidden_states.shape
        residual = hidden_states
        query = self.module.to_q(hidden_states[mask].contiguous())

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        kv = self.to_kv(encoder_hidden_states[mask].contiguous())
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
        # if self.cross_attn_output is None:
        #     self.cross_attn_output = hidden_states
        # else:
        #     self.cross_attn_output[mask] = hidden_states
        # hidden_states = self.cross_attn_output
        hidden_states = self.cross_attn_output.update_and_return(input_indices, hidden_states, mask)
        if self.module.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / self.module.rescale_output_factor
        # torch.cuda.synchronize()
        # self.cross_attn_time += (time.time() - start)
        return hidden_states

class PatchSelfAttention(PatchAttention):
    def __init__(self, module: Attention):
        super(PatchSelfAttention, self).__init__(module)
        self.self_attention_time = 0
        self.streams = []
        for _ in range(3):
            self.streams.append(torch.cuda.Stream())
        self.self_attention_output = CacheManager()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        is_sliced: bool = False,
        latent_offset: list = None,
        resolution_offset: list = None,
        mask: list = None,
        input_indices: list = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # torch.cuda.synchronize()
        # start = time.time()
        # print(mask)
        shape = hidden_states.shape
        if len(shape) == 3:
            b, sl, q_c = hidden_states.shape
        else:
            b, h, w, q_c = hidden_states.shape[-1]
            sl = h * w
        
        residual = hidden_states
        bs = 0
        encoder_hidden_states = hidden_states
        # print(encoder_hidden_states.shape)
        kv = self.to_kv(encoder_hidden_states, input_indices=input_indices, mask=mask)
        # print(kv.shape)
        c = kv.shape[-1]
        query = self.module.to_q(hidden_states, input_indices=input_indices, mask=mask)
        if is_sliced:
        # if False:
            states = []
            for resolution_index in range(len(resolution_offset) - 1):
                if mask[latent_offset[resolution_offset[resolution_index]] : latent_offset[resolution_offset[resolution_index + 1]]].sum() == 0:
                    continue

                batch_size = latent_offset[resolution_offset[resolution_index + 1]] - latent_offset[resolution_offset[resolution_index]]
                sparse_ratio = mask[latent_offset[resolution_offset[resolution_index]] : latent_offset[resolution_offset[resolution_index + 1]]].sum() / batch_size
                if sparse_ratio < 0.5:
                    for latent_index in range(resolution_offset[resolution_index], resolution_offset[resolution_index + 1]):
                        query_mask = mask[latent_offset[latent_index] : latent_offset[latent_index + 1]]
                        if query_mask.sum() == 0:
                            continue
                        ker_per_resolution = kv[latent_offset[latent_index] : latent_offset[latent_index + 1]].view(1, -1, c)
                        query_per_resolution = query[latent_offset[latent_index] : latent_offset[latent_index + 1]][query_mask].view(1, -1, q_c)
                        key, value = torch.split(ker_per_resolution, ker_per_resolution.shape[-1] // 2, dim=-1)
                        query_per_resolution = self.module.head_to_batch_dim(query_per_resolution).contiguous()
                        key = self.module.head_to_batch_dim(key).contiguous()
                        value = self.module.head_to_batch_dim(value).contiguous()
                        result = xformers.ops.memory_efficient_attention(query_per_resolution, key, value)
                        result = result.to(query.dtype)
                        result = self.module.batch_to_head_dim(result)
                        states.append(result.view(int(query_mask.sum()), -1, q_c))
                else:
                    base = 0
                    latent_size = resolution_offset[resolution_index + 1] - resolution_offset[resolution_index]
                    patch_per_latent = batch_size // latent_size

                    c = kv.shape[-1]

                    ker_per_resolution = kv[latent_offset[resolution_offset[resolution_index]] : latent_offset[resolution_offset[resolution_index + 1]]].view(latent_size, -1, c)
                    query_per_resolution = query[latent_offset[resolution_offset[resolution_index]] : latent_offset[resolution_offset[resolution_index + 1]]].view(latent_size, -1, q_c)
                    inner_dim = query_per_resolution.shape[1]

                    key, value = torch.split(ker_per_resolution, ker_per_resolution.shape[-1] // 2, dim=-1)

                    query_per_resolution = self.module.head_to_batch_dim(query_per_resolution).contiguous()
                    # print(query_per_resolution.shape)
                    bs = key.shape[1]
                    key = self.module.head_to_batch_dim(key).contiguous()
                    value = self.module.head_to_batch_dim(value).contiguous()
                    
                    result = xformers.ops.memory_efficient_attention(query_per_resolution, key, value)

                    result = result.to(query.dtype)
                    result = self.module.batch_to_head_dim(result)
                    # result = result[mask]
                    # result = self.module.to_out[0](result)
                    states.append(result.view(batch_size, -1, q_c)[mask[latent_offset[resolution_offset[resolution_index]] : latent_offset[resolution_offset[resolution_index + 1]]]])
                
            output = torch.cat(states, dim=0)

        else:

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
            output = self.module.batch_to_head_dim(hidden_states)
        
        output = self.module.to_out[0](output)
        # dropout
        output = self.module.to_out[1](output)
        if mask is not None:
            hidden_states = self.self_attention_output.update_and_return(input_indices, output, mask)
        else:
            hidden_states = output
        if self.module.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / self.module.rescale_output_factor
        # torch.cuda.synchronize()
        # self.self_attention_time += (time.time() - start)
        return hidden_states
    
class PatchSD3Attention(PatchAttention):
    def __init__(self, module: Attention):
        super(PatchSD3Attention, self).__init__(module)
        self.self_attention_time = 0
        self.output = CacheManager()
        self.encoder_output = CacheManager()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        is_sliced: bool = False,
        latent_offset: list = None,
        resolution_offset: list = None,
        mask: list = None,
        input_indices: list = None,
        encoder_indices: list = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        # `sample` projections.
        mask = mask.copy()
        query = self.module.to_q(hidden_states)
        self_kv = self.to_kv(hidden_states)
        key, value = torch.split(self_kv, self_kv.shape[-1] // 2, dim=-1)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.module.heads

        query = query.reshape(batch_size, -1, self.module.heads, head_dim)
        key = key.reshape(batch_size, -1, self.module.heads, head_dim)
        value = value.reshape(batch_size, -1, self.module.heads, head_dim)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_mask = np.zeros(encoder_hidden_states.shape[0]) > 0.5
            encoder_hidden_states_query_proj = self.module.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = self.module.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.module.add_v_proj(encoder_hidden_states)
            if is_sliced:
                encoder_bs = len(latent_offset) - 1
            else:
                encoder_bs = batch_size
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.reshape(
                encoder_bs, -1, self.module.heads, head_dim
            )
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.reshape(
                encoder_bs, -1, self.module.heads, head_dim
            )
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.reshape(
                encoder_bs, -1, self.module.heads, head_dim
            )
            _,sl, _, _ = encoder_hidden_states_query_proj.shape
            # query = torch.cat([query.transpose(1, 2), encoder_hidden_states_query_proj.transpose(1, 2)], dim=2)
            # key = torch.cat([key.transpose(1, 2), encoder_hidden_states_key_proj.transpose(1, 2)], dim=2)
            # value = torch.cat([value.transpose(1, 2), encoder_hidden_states_value_proj.transpose(1, 2)], dim=2)
        skip = False
        if is_sliced:
            latent_states = []
            encoder_states = []
            base = 0
            for resolution_index in range(len(resolution_offset) - 1):
                start = latent_offset[resolution_offset[resolution_index]]
                end = latent_offset[resolution_offset[resolution_index + 1]]
                latent_size = resolution_offset[resolution_index + 1] - resolution_offset[resolution_index]
                if mask[start : end].sum() == 0:
                    skip = True
                    base += latent_size
                    continue
                bs = end - start
                sparse_ratio = mask[start : end].sum() / bs
                if sparse_ratio <= 1/16 and encoder_hidden_states is None:
                    # print(sparse_ratio)
                    skip = True
                    for latent_index in range(resolution_offset[resolution_index], resolution_offset[resolution_index + 1]):
                        latent_start = latent_offset[latent_index]
                        latent_end = latent_offset[latent_index + 1]
                        query_mask = mask[latent_start : latent_end]
                        if query_mask.sum() == 0:
                            base += 1
                            continue
                        query_per_request = query[latent_start : latent_end][query_mask].view(1, -1, self.module.heads, head_dim).transpose(1, 2)
                        key_per_request = key[latent_start : latent_end].view(1, -1, self.module.heads, head_dim).transpose(1, 2)
                        value_per_request = value[latent_start : latent_end].view(1, -1, self.module.heads, head_dim).transpose(1, 2)
                        if self.module.norm_q is not None:
                            query_per_request = self.module.norm_q(query_per_request)
                        if self.module.norm_k is not None:
                            key_per_request = self.module.norm_k(key_per_request)
                        result = F.scaled_dot_product_attention(query_per_request, key_per_request, value_per_request, dropout_p=0.0, is_causal=False)
                        result = result.to(query.dtype).transpose(1, 2)
                        latent_states.append(result.reshape(int(query_mask.sum()), -1, inner_dim))
                        base += 1
                else:
                    key_per_resolution = key[start : end].view(latent_size, -1, self.module.heads, head_dim).transpose(1, 2)
                    value_per_resolution = value[start : end].view(latent_size, -1, self.module.heads, head_dim).transpose(1, 2)
                    query_per_resolution = query[start : end].view(latent_size, -1, self.module.heads, head_dim).transpose(1, 2)
                    if self.module.norm_q is not None:
                        query_per_resolution = self.module.norm_q(query_per_resolution)
                    if self.module.norm_k is not None:
                        key_per_resolution = self.module.norm_k(key_per_resolution)
                    if encoder_hidden_states is not None:
                        encoder_hidden_states_query_proj_per_resolution = encoder_hidden_states_query_proj[base : base + latent_size].transpose(1, 2)
                        
                        encoder_hidden_states_key_proj_per_resolution = encoder_hidden_states_key_proj[base : base + latent_size].transpose(1, 2)
                        
                        encoder_hidden_states_value_proj_per_resolution = encoder_hidden_states_value_proj[base : base + latent_size].transpose(1, 2)
                        value_per_resolution = torch.cat([value_per_resolution, encoder_hidden_states_value_proj_per_resolution], dim=2)
                        if self.module.norm_added_q is not None:
                            encoder_hidden_states_query_proj_per_resolution = self.module.norm_added_q(encoder_hidden_states_query_proj_per_resolution)
                        if self.module.norm_added_k is not None:
                            encoder_hidden_states_key_proj_per_resolution = self.module.norm_added_k(encoder_hidden_states_key_proj_per_resolution)
                        query_per_resolution = torch.cat([query_per_resolution, encoder_hidden_states_query_proj_per_resolution], dim=2)
                        key_per_resolution = torch.cat([key_per_resolution, encoder_hidden_states_key_proj_per_resolution], dim=2)
                    
                    result = F.scaled_dot_product_attention(query_per_resolution, key_per_resolution, value_per_resolution, dropout_p=0.0, is_causal=False)

                    result = result.to(query.dtype)
                    output = result.transpose(1, 2)
                    # result = result[mask]
                    # result = self.module.to_out[0](result)
                    if encoder_hidden_states is not None:
                        output, encoder_output = (
                            output[:, : residual.shape[1] * int(bs // latent_size)],
                            output[:, residual.shape[1] * int(bs // latent_size) :],
                        )
                        encoder_states.append(encoder_output.reshape(latent_size, -1, inner_dim))
                        encoder_mask[base: base + latent_size] = True
                    mask[start : end] = True
                    base += latent_size
                    latent_states.append(output.reshape(bs, -1, inner_dim))
                    
                
            output = torch.cat(latent_states, dim=0)
            if encoder_hidden_states is not None:
                encoder_output = torch.cat(encoder_states, dim=0)
                if not self.module.context_pre_only:
                    encoder_output = self.module.to_add_out(encoder_output)
        else:
            query = query[ : ].view(batch_size, -1, self.module.heads, head_dim).transpose(1, 2)
            key = key[ : ].view(batch_size, -1, self.module.heads, head_dim).transpose(1, 2)
            value = value[ : ].view(batch_size, -1, self.module.heads, head_dim).transpose(1, 2)
            if self.module.norm_q is not None:
                query = self.module.norm_q(query)
            if self.module.norm_k is not None:
                key = self.module.norm_k(key)
            if encoder_hidden_states is not None:
                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.transpose(1, 2)
                if self.module.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.module.norm_added_q(encoder_hidden_states_query_proj)
                if self.module.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.module.norm_added_k(encoder_hidden_states_key_proj)
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
                
            
            output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            # output = xformers.ops.memory_efficient_attention(query, key, value)
            output = output.transpose(1, 2).reshape(batch_size, -1, self.module.heads * head_dim)
            output = output.to(query.dtype)

            if encoder_hidden_states is not None:
                # Split the attention outputs.
                output, encoder_output = (
                    output[:, : residual.shape[1]],
                    output[:, residual.shape[1] :],
                )
                if not self.module.context_pre_only:
                    encoder_output = self.module.to_add_out(encoder_output)
                

        # linear proj
        # print(encoder_output.shape)
        output = self.module.to_out[0](output)
        # dropout
        output = self.module.to_out[1](output)
        if mask is not None:
            hidden_states = self.output.update_and_return(input_indices, output, mask, skip)
            if encoder_hidden_states is not None:
                encoder_hidden_states = self.encoder_output.update_and_return(encoder_indices, encoder_output, encoder_mask, skip)
        else:
            hidden_states = output

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
        
