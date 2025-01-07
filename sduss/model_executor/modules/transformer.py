# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from .base_module import BaseModule
from diffusers.models.transformers.transformer_2d import Transformer2DModel, Transformer2DModelOutput
from diffusers.models.attention import BasicTransformerBlock, JointTransformerBlock
from typing import Optional, List, Tuple, Union, Dict, Any

class PatchTransformer2DModel(BaseModule):
    def __init__(
        self,
        module: Transformer2DModel
    ):
        super().__init__(module)
    
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        patch_map: dict = None,
        cross_attention_kwargs=None,
        return_dict: bool = True,
        latent_offset: dict = None,
        padding_idx: dict = None,
        is_sliced:bool = False,
        resolution_offset: dict = None,
        mask: list = None,
        input_indices: list = None,
    ):
        # 1. Input
        if self.module.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.module.norm(hidden_states, is_sliced=is_sliced, is_fused=False, latent_offset=latent_offset["cuda"], padding_idx=padding_idx["cuda"], patch_map=patch_map["cuda"])
            if not self.module.use_linear_projection:
                hidden_states = self.module.proj_in(hidden_states, is_sliced=is_sliced, input_indices=input_indices, mask=mask)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, -1, inner_dim)
                hidden_states = self.module.proj_in(hidden_states, input_indices=input_indices, mask=mask)
        elif self.module.is_input_vectorized:
            hidden_states = self.module.latent_image_embedding(hidden_states)
        elif self.module.is_input_patches:
            hidden_states = self.module.pos_embed(hidden_states)

        # if self.module.caption_projection is not None:
        #     batch_size = hidden_states.shape[0]
        #     encoder_hidden_states = self.module.caption_projection(encoder_hidden_states)
        #     encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # 2. Blocks
        for block in self.module.transformer_blocks:
            
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                latent_offset=latent_offset,
                is_sliced=is_sliced,
                resolution_offset=resolution_offset,
                mask=mask,
                input_indices=input_indices,
            )            

        # 3. Output
        if self.module.is_input_continuous:
            if not self.module.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.module.proj_out(hidden_states, is_sliced=is_sliced, input_indices=input_indices, mask=mask)
            else:
                hidden_states = self.module.proj_out(hidden_states, input_indices=input_indices, mask=mask)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.module.is_input_vectorized:
            hidden_states = self.module.norm_out(hidden_states)
            logits = self.module.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.module.is_input_patches:
            # TODO: cleanup!
            conditioning = self.module.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.module.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.module.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.module.proj_out_2(hidden_states)

            # unpatchify
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output

class PatchBasicTransformerBlock(BaseModule):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        module: BasicTransformerBlock
    ):
        super().__init__(module)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        latent_offset: dict = None,
        is_sliced: bool = False,
        resolution_offset: dict = None,
        mask:list = None,
        input_indices: list = None,
    ):
        batch = hidden_states.shape[0]
        if self.module.norm_type == "ada_norm":
            norm_hidden_states = self.module.norm1(hidden_states, timestep)
        elif self.module.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.module.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.module.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.module.norm1(hidden_states)
        elif self.module.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.module.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.module.pos_embed is not None:
            norm_hidden_states = self.module.pos_embed(norm_hidden_states)


        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.module.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.module.only_cross_attention else None,
            attention_mask=attention_mask,
            latent_offset=latent_offset["cpu"],
            is_sliced=is_sliced,
            resolution_offset=resolution_offset["cpu"],
            mask=mask,
            input_indices=input_indices,
            **cross_attention_kwargs,
        )
        # hidden_states = hidden_states.reshape(batch, height * width, -1)
        if self.module.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.module.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        if self.module.attn2 is not None:
            if self.module.norm_type == "ada_norm":
                norm_hidden_states = self.module.norm2(hidden_states, timestep)
            elif self.module.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.module.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.module.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.module.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            # 2. Cross-Attention
            attn_output = self.module.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                latent_offset=latent_offset["cpu"],
                is_sliced=is_sliced,
                mask=mask,
                input_indices=input_indices,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states


        if self.module.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.module.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.module.norm_type == "ada_norm_single":
            norm_hidden_states = self.module.norm3(hidden_states)

        if self.module.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.module.norm_type == "ada_norm_single":
            norm_hidden_states = self.module.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self.module._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.module.ff, norm_hidden_states, self.module._chunk_dim, self.module._chunk_size)
        else:
            ff_output = self.module.ff(norm_hidden_states, mask=mask)

        if self.module.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.module.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

class PatchJointTransformerBlock(BaseModule):
    def __init__(
        self,
        module: JointTransformerBlock,
    ):
        super().__init__(module)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        temb_latent: torch.FloatTensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        is_sliced: bool = False,
        latent_offset: list = None,
        resolution_offset: list = None,
        state_mask: list = None,
        input_indices: list = None,
        encoder_indices: list = None,
    ):
        

        joint_attention_kwargs = joint_attention_kwargs or {}
        if self.module.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.module.norm1(
                hidden_states, emb=temb_latent
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.module.norm1(hidden_states, emb=temb_latent)

        if self.module.context_pre_only:
            norm_encoder_hidden_states = self.module.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.module.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.module.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            is_sliced=is_sliced,
            latent_offset=latent_offset["cpu"],
            resolution_offset=resolution_offset["cpu"],
            mask=state_mask,
            input_indices=input_indices,
            encoder_indices=encoder_indices,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.module.use_dual_attention:
            attn_output2 = self.module.attn2(hidden_states=norm_hidden_states2, 
                                            is_sliced=is_sliced,
                                            latent_offset=latent_offset["cpu"],
                                            resolution_offset=resolution_offset["cpu"],
                                            mask=state_mask,
                                            input_indices=input_indices,
                                            encoder_indices=encoder_indices,
                                            **joint_attention_kwargs)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.module.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self.module._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.module.ff, norm_hidden_states, self.module._chunk_dim, self.module._chunk_size)
        else:
            ff_output = self.module.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.module.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.module.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self.module._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.module.ff_context, norm_encoder_hidden_states, self.module._chunk_dim, self.module._chunk_size
                )
            else:
                context_ff_output = self.module.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states

