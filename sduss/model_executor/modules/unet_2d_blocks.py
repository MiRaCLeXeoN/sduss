from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn, CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D
from diffusers.utils.torch_utils import apply_freeu
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Dict, Any
from .base_module import BaseModule
from .cache_manager import CacheManager

class PatchUNetMidBlock2DCrossAttn(BaseModule):
    def __init__(
        self,
        module: UNetMidBlock2DCrossAttn
    ):
        super().__init__(module)
        self.has_cross_attention = True
        # self.input = None
        self.output = CacheManager()
        # self.previous_mask = None
        self.input = CacheManager()
        # self.output = CacheManager()

    
    def forward(
        self, hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        latent_offset: dict = None,
        padding_idx: dict = None,
        is_sliced:bool = False,
        patch_map:dict = None,
        resolution_offset: dict = None,
        index: int = 0,
        timestep: torch.FloatTensor = None,
        total_blocks: int = 0,
        input_indices: list = None,
    ):
        mask = self.input.get_mask(input_indices, hidden_states, total_blocks, timestep, False)
        if mask.sum() != 0:
            hidden_states = self.module.resnets[0](hidden_states, temb, is_sliced=is_sliced, input_indices=input_indices, patch_map=patch_map, latent_offset=latent_offset, padding_idx=padding_idx, mask=mask)
            for attn, resnet in zip(self.module.attentions, self.module.resnets[1:]):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    is_sliced=is_sliced,
                    patch_map=patch_map,
                    padding_idx=padding_idx,
                    latent_offset=latent_offset,
                    resolution_offset=resolution_offset,
                    mask=mask,
                    input_indices=input_indices,
                ).sample
                hidden_states = resnet(hidden_states, temb, mask=mask, input_indices=input_indices, is_sliced=is_sliced, patch_map=patch_map, latent_offset=latent_offset, padding_idx=padding_idx)
        # else:
        hidden_states = self.output.save_and_get_block_states(input_indices, hidden_states, mask)

        return hidden_states

class PatchCrossAttnDownBlock2D(BaseModule):
    def __init__(
        self,
        module: CrossAttnDownBlock2D
    ):
        super().__init__(module)
        self.has_cross_attention = True
        self.input = CacheManager()
        # self.input = None
        self.output = CacheManager()
        self.out_states = CacheManager()
        self.previous_mask = None
        output_num = 0
        for resnet, attn in zip(self.module.resnets, self.module.attentions):
            output_num += 1
        if self.module.downsamplers is not None:
            for downsampler in self.module.downsamplers:
                output_num += 1
        self.output_num = output_num
    
    def forward(
        self, hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        additional_residuals: Optional[torch.FloatTensor] = None,
        latent_offset: dict = None,
        padding_idx: dict = None,
        is_sliced:bool = False,
        patch_map: dict = None,
        resolution_offset: dict = None,
        index: int = 0,
        timestep: torch.FloatTensor = None,
        total_blocks: int = 0,
        input_indices: list = None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()
        mask = self.input.get_mask(input_indices, hidden_states, total_blocks, timestep, False)
        if mask.sum() != 0:
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
                    hidden_states = resnet(hidden_states, temb, mask=mask, input_indices=input_indices, is_sliced=is_sliced, patch_map=patch_map, latent_offset=latent_offset, padding_idx=padding_idx)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        is_sliced=is_sliced,
                        patch_map=patch_map,
                        padding_idx=padding_idx,
                        latent_offset=latent_offset,
                        resolution_offset=resolution_offset,
                        mask=mask,
                        input_indices=input_indices,
                    ).sample

                output_states += (hidden_states,)
            if self.module.downsamplers is not None:
                for downsampler in self.module.downsamplers:
                    hidden_states = downsampler(hidden_states, mask=mask, input_indices=input_indices, is_sliced=is_sliced, padding_idx=padding_idx)

                output_states += (hidden_states,)
        # else:
        hidden_states = self.output.save_and_get_block_states(input_indices, hidden_states, mask)
        output_states = self.out_states.save_and_get_block_tupple(input_indices, output_states, mask, self.output_num)
        # self.output = hidden_states
        # self.out_states = output_states
        return hidden_states, output_states

class PatchDownBlock2D(BaseModule):
    def __init__(
        self,
        module: DownBlock2D
    ):
        super().__init__(module)
        self.input = CacheManager()
        # self.input = None
        self.output = CacheManager()
        self.out_states = CacheManager()
        self.previous_mask = None
        output_num = 0
        for resnet in self.module.resnets:
            output_num += 1
        if self.module.downsamplers is not None:
            for downsampler in self.module.downsamplers:
                output_num += 1
        self.output_num = output_num

    def forward(self, hidden_states, temb=None, 
                index: int = 0,
        timestep: torch.FloatTensor = None,
        total_blocks: int = 0,
        input_indices: list = None,
                latent_offset: dict = None,
                padding_idx: dict = None,
                patch_map: dict = None,
                is_sliced:bool = False):
        output_states = ()
        mask = self.input.get_mask(input_indices, hidden_states, total_blocks, timestep, False)
        if mask.sum() > 0:
            for resnet in self.module.resnets:
                if self.module.training and self.module.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                else:
                    hidden_states = resnet(hidden_states, temb, mask=mask, input_indices=input_indices, is_sliced=is_sliced, patch_map=patch_map, latent_offset=latent_offset,padding_idx=padding_idx)

                output_states += (hidden_states,)
            if self.module.downsamplers is not None:
                for downsampler in self.module.downsamplers:
                    hidden_states = downsampler(hidden_states, mask=mask, input_indices=input_indices, is_sliced=is_sliced, padding_idx=padding_idx)

                output_states += (hidden_states,)
        # else:
        hidden_states = self.output.save_and_get_block_states(input_indices, hidden_states, mask)
        output_states = self.out_states.save_and_get_block_tupple(input_indices, output_states, mask, self.output_num)
        # self.output = hidden_states
        # self.out_states = output_states

        

        return hidden_states, output_states

class PatchCrossAttnUpBlock2D(BaseModule):
    def __init__(
        self,
        module: CrossAttnUpBlock2D
    ):
        super().__init__(module)
        self.input = CacheManager()
        # self.input = None
        self.output = CacheManager()
        self.res_tuple = None
        self.has_cross_attention = True
        self.previous_mask = None
    
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
        latent_offset: dict = None,
        padding_idx: dict = None,
        is_sliced:bool = False,
        patch_map: dict = None,
        resolution_offset: dict = None,
        index: int = 0,
        timestep: torch.FloatTensor = None,
        total_blocks: int = 0,
        input_indices: list = None,
    ):
        is_freeu_enabled = (
            getattr(self.module, "s1", None)
            and getattr(self.module, "s2", None)
            and getattr(self.module, "b1", None)
            and getattr(self.module, "b2", None)
        )
        mask = self.input.get_mask(input_indices, hidden_states, total_blocks, timestep, True, res_hidden_states_tuple)
        
        # TODO(Patrick, William) - attention mask is not used
        if mask.sum() > 0:
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
                    hidden_states = resnet(hidden_states, temb, mask=mask, input_indices=input_indices, is_sliced=is_sliced, patch_map=patch_map, latent_offset=latent_offset,padding_idx=padding_idx)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        is_sliced=is_sliced,
                        patch_map=patch_map,
                        padding_idx=padding_idx,
                        latent_offset=latent_offset,
                        mask=mask,
                        input_indices=input_indices,
                        resolution_offset=resolution_offset,
                    ).sample
            if self.module.upsamplers is not None:
                for upsampler in self.module.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size, mask=mask, input_indices=input_indices, is_sliced=is_sliced, padding_idx=padding_idx)

        # else:
        hidden_states = self.output.save_and_get_block_states(input_indices, hidden_states, mask)
        # self.output = hidden_states

        # exit(0)
        return hidden_states

class PatchUpBlock2D(BaseModule):
    def __init__(
        self,
        module: UpBlock2D
    ):
        super().__init__(module)
        self.input = CacheManager()
        self.res_tuple = None
        # self.input = None
        self.output = CacheManager()
        self.previous_mask = None
    
    def forward(self, hidden_states, 
                res_hidden_states_tuple, 
                index: int = 0,
                input_indices: list = None,
        timestep: torch.FloatTensor = None,
        total_blocks: int = 0,
                temb=None, upsample_size=None,
                latent_offset: dict = None,
                padding_idx: dict = None,
                patch_map: dict = None,
                is_sliced:bool = False):
        
        is_freeu_enabled = (
            getattr(self.module, "s1", None)
            and getattr(self.module, "s2", None)
            and getattr(self.module, "b1", None)
            and getattr(self.module, "b2", None)
        )
        batch_size = hidden_states.shape[0]
        # mask = np.ones(batch_size) > 0.5
        mask = self.input.get_mask(input_indices, hidden_states, total_blocks, timestep, True, res_hidden_states_tuple)
        if mask.sum() > 0:
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
                    hidden_states = resnet(hidden_states, temb, mask=mask, input_indices=input_indices, is_sliced=is_sliced, patch_map=patch_map, latent_offset=latent_offset, padding_idx=padding_idx)
            if self.module.upsamplers is not None:
                for upsampler in self.module.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size, input_indices=input_indices, mask=mask, is_sliced=is_sliced, padding_idx=padding_idx)

        # else:
        hidden_states = self.output.save_and_get_block_states(input_indices, hidden_states, mask)
        # self.output = hidden_states

        
        return hidden_states