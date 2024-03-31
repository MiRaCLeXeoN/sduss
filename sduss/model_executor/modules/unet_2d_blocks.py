from typing import Optional, List, Tuple, Union, Dict, Any

import torch
import torch.nn as nn

from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn, CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D
from diffusers.utils.torch_utils import apply_freeu

from .base_module import BaseModule

class PatchUNetMidBlock2DCrossAttn(BaseModule):
    def __init__(
        self,
        module: UNetMidBlock2DCrossAttn
    ):
        super().__init__(module)
        self.has_cross_attention = True
    
    def forward(
        self, hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        image_offset: list = [],
        padding_idx: list = [],
        is_sliced:bool = False,
        resolution_offset: list = [],
    ):
        hidden_states = self.module.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.module.attentions, self.module.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                # is_sliced=is_sliced,
                # image_offset=image_offset,
                # resolution_offset=resolution_offset,
            ).sample
            hidden_states = resnet(hidden_states, temb, is_sliced=is_sliced, padding_idx=padding_idx)

        return hidden_states

class PatchCrossAttnDownBlock2D(BaseModule):
    def __init__(
        self,
        module: CrossAttnDownBlock2D
    ):
        super().__init__(module)
        self.has_cross_attention = True
    
    def forward(
        self, hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        additional_residuals: Optional[torch.FloatTensor] = None,
        image_offset: list = [],
        padding_idx: list = [],
        is_sliced:bool = False,
        resolution_offset: list = [],
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        for resnet, attn in zip(self.module.resnets, self.module.attentions):
            if self.module.training and self.module.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, is_sliced=is_sliced, padding_idx=padding_idx)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # is_sliced=is_sliced,
                    # image_offset=image_offset,
                    # resolution_offset=resolution_offset,
                ).sample

            output_states += (hidden_states,)

        if self.module.downsamplers is not None:
            for downsampler in self.module.downsamplers:
                hidden_states = downsampler(hidden_states, is_sliced=is_sliced, padding_idx=padding_idx)

            output_states += (hidden_states,)

        return hidden_states, output_states

class PatchDownBlock2D(BaseModule):
    def __init__(
        self,
        module: DownBlock2D
    ):
        super().__init__(module)

    def forward(self, hidden_states, temb=None, 
                image_offset: list = [],
                padding_idx: list = [],
                is_sliced:bool = False):
        output_states = ()

        for resnet in self.module.resnets:
            if self.module.training and self.module.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb, is_sliced=is_sliced, padding_idx=padding_idx)

            output_states += (hidden_states,)

        if self.module.downsamplers is not None:
            for downsampler in self.module.downsamplers:
                hidden_states = downsampler(hidden_states, is_sliced=is_sliced, padding_idx=padding_idx)

            output_states += (hidden_states,)

        return hidden_states, output_states

class PatchCrossAttnUpBlock2D(BaseModule):
    def __init__(
        self,
        module: CrossAttnUpBlock2D
    ):
        super().__init__(module)
        self.has_cross_attention = True
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        image_offset: list = [],
        padding_idx: list = [],
        is_sliced:bool = False,
        resolution_offset: list = [],
    ):
        is_freeu_enabled = (
            getattr(self.module, "s1", None)
            and getattr(self.module, "s2", None)
            and getattr(self.module, "b1", None)
            and getattr(self.module, "b2", None)
        )

        # TODO(Patrick, William) - attention mask is not used
        for resnet, attn in zip(self.module.resnets, self.module.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.module.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.module.s1,
                    s2=self.module.s2,
                    b1=self.module.b1,
                    b2=self.module.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.module.training and self.module.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, is_sliced=is_sliced, padding_idx=padding_idx)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # is_sliced=is_sliced,
                    # image_offset=image_offset,
                    # resolution_offset=resolution_offset,
                ).sample

        if self.module.upsamplers is not None:
            for upsampler in self.module.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, is_sliced=is_sliced, padding_idx=padding_idx)

        return hidden_states

class PatchUpBlock2D(BaseModule):
    def __init__(
        self,
        module: UpBlock2D
    ):
        super().__init__(module)
    
    def forward(self, hidden_states, 
                res_hidden_states_tuple, 
                temb=None, upsample_size=None,
                image_offset: list = [],
                padding_idx: list = [],
                is_sliced:bool = False):
        
        is_freeu_enabled = (
            getattr(self.module, "s1", None)
            and getattr(self.module, "s2", None)
            and getattr(self.module, "b1", None)
            and getattr(self.module, "b2", None)
        )
        for resnet in self.module.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.module.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.module.s1,
                    s2=self.module.s2,
                    b1=self.module.b1,
                    b2=self.module.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.module.training and self.module.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb, is_sliced=is_sliced, padding_idx=padding_idx)

        if self.module.upsamplers is not None:
            for upsampler in self.module.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, is_sliced=is_sliced, padding_idx=padding_idx)

        return hidden_states