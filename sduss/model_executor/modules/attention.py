import torch
from diffusers.models.attention_processor import Attention
from torch import nn
from torch.nn import functional as F
import xformers
import xformers.ops
# from distrifuser.modules.base_module import BaseModule
# from distrifuser.utils import DistriConfig
from .resnet import SplitLinear
from .base_module import BaseModule
from .cache_manager import CacheManager
import math
import time
from torch.utils.cpp_extension import load


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
        encoder_hidden_states: torch.FloatTensor or None = None,
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
        query = self.module.to_q(hidden_states[mask])

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        kv = self.to_kv(encoder_hidden_states[mask])
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
        encoder_hidden_states: torch.FloatTensor or None = None,
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
                if sparse_ratio < 0.8:
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
    
