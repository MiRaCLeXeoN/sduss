import math
import time

from typing import Optional, Union

import torch.nn.functional as F
import torch

from diffusers.models.resnet import  ResnetBlock2D
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn, CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from torch import distributed as dist, nn

from .resnet import SplitModule, SplitConv, SplitGroupnorm, SplitLinear, PatchConv, PatchUpsample2D, PatchDownsample2D, PatchResnetBlock2D
from .base_module import BaseModel, BaseModule
from .attention import  PatchCrossAttention, PatchSelfAttention
from .groupnorm import PatchGroupNorm
from .transformer import PatchTransformer2DModel, PatchBasicTransformerBlock
from .unet_2d_blocks import PatchUNetMidBlock2DCrossAttn, PatchCrossAttnDownBlock2D, PatchDownBlock2D, PatchCrossAttnUpBlock2D, PatchUpBlock2D

class PatchUNet(BaseModel):  # for Patch Parallelism
    def __init__(self, model: UNet2DConditionModel):
        # assert isinstance(model, UNet2DConditionModel)
        self.total_time = 0
        for name, module in model.named_modules():
            if isinstance(module, BaseModule):
                continue
            if isinstance(module, SplitModule):
                continue
            
            for subname, submodule in module.named_children():
                if isinstance(submodule, nn.Conv2d):
                    kernel_size = submodule.kernel_size
                    conv_module = SplitConv(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, 0 if subname=="conv_in" else submodule.padding, submodule.dilation, submodule.groups, dtype=torch.float16)
                    conv_module.weight.data.copy_(submodule.weight.data)
                    if submodule.bias is not None:
                        conv_module.bias.data.copy_(submodule.bias.data)
                    wrapped_submodule = PatchConv(conv_module)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, nn.Linear):
                    linear_module = SplitLinear(submodule.in_features, submodule.out_features, dtype=torch.float16)
                    linear_module.weight.data.copy_(submodule.weight.data)
                    if submodule.bias is not None:
                        linear_module.bias.data.copy_(submodule.bias.data)
                    setattr(module, subname, linear_module)
                elif isinstance(submodule, Attention):
                    if subname == "attn1":  #s self attention
                        wrapped_submodule = PatchSelfAttention(submodule)
                    else:  # cross attention
                        assert subname == "attn2"
                        wrapped_submodule = PatchCrossAttention(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, nn.GroupNorm):
                    groupnorm_module = SplitGroupnorm(submodule.num_groups, submodule.num_channels, submodule.eps, submodule.affine, dtype=torch.float16)
                    groupnorm_module.weight.data.copy_(submodule.weight.data)
                    if submodule.bias is not None:
                        groupnorm_module.bias.data.copy_(submodule.bias.data)
                    wrapped_submodule = PatchGroupNorm(groupnorm_module)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, Upsample2D):
                    wrapped_submodule = PatchUpsample2D(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, Downsample2D):
                    wrapped_submodule = PatchDownsample2D(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, ResnetBlock2D):
                    wrapped_submodule = PatchResnetBlock2D(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, Transformer2DModel):
                    wrapped_submodule = PatchTransformer2DModel(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, BasicTransformerBlock):
                    wrapped_submodule = PatchBasicTransformerBlock(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, UNetMidBlock2DCrossAttn):
                    wrapped_submodule = PatchUNetMidBlock2DCrossAttn(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, CrossAttnDownBlock2D):
                    wrapped_submodule = PatchCrossAttnDownBlock2D(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, DownBlock2D):
                    wrapped_submodule = PatchDownBlock2D(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, CrossAttnUpBlock2D):
                    wrapped_submodule = PatchCrossAttnUpBlock2D(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, UpBlock2D):
                    wrapped_submodule = PatchUpBlock2D(submodule)
                    setattr(module, subname, wrapped_submodule)
                # elif isinstance(submodule, Upsample2D):
                    # wrapped_submodule = PatchUpsample2D(submodule)
                    # setattr(module, subname, wrapped_submodule)
        self.diff = None
        super(PatchUNet, self).__init__(model)
        # print(self.config)


    def split_sample(self, samples, patch_size, input_indices):
        latent_offset = list()
        patch_map = list()
        latent_offset.append(0)
        resolution_offset = list()
        resolution_offset.append(0)
        padding_idx = list()
        new_sample = list()
        indices = list()
        for resolution, res_sample in samples.items():
            resolution = int(resolution)
            patch_on_height = (resolution // patch_size)
            patch_on_width = (resolution // patch_size)
            latent_patch_size = int(patch_size // 8)
            if res_sample is None or res_sample.shape[0] == 0:
                continue
            index = 0
            for sample in res_sample:
                latent_offset.append(latent_offset[-1] + patch_on_width ** 2)
                sample = torch.nn.functional.pad(sample, (1, 1, 1, 1), "constant", 0).unsqueeze(0)
                
                for h in range((patch_on_height)):
                    for w in range((patch_on_width)):
                        paddings = torch.empty(4, device=sample.device, dtype=torch.int32)
                        # paddings = [None] * 4
                        if (patch_on_height) == 1:
                            paddings[0] = -1
                            paddings[1] = -1
                            paddings[2] = -1
                            paddings[3] = -1
                            new_sample.append(sample[:, :, h * latent_patch_size : (h + 1) * latent_patch_size + 2, w * latent_patch_size : (w + 1) * latent_patch_size + 2])
                            patch_map.append(len(latent_offset)-1)
                            padding_idx.append(paddings)
                            continue
                        if w == 0:
                            paddings[1] = -1
                            paddings[3] = len(new_sample) + 1
                        elif w == (patch_on_width) - 1:
                            paddings[1] = len(new_sample) - 1
                            paddings[3] = -1
                        else:
                            paddings[1] = len(new_sample) - 1
                            paddings[3] = len(new_sample) + 1
                        if h == 0:
                            paddings[0] = -1
                            paddings[2] = len(new_sample) + (patch_on_height)
                        elif h == (patch_on_height) - 1:
                            paddings[0] = len(new_sample) - (patch_on_height)
                            paddings[2] = -1
                        else:
                            paddings[0] = len(new_sample) - (patch_on_height)
                            paddings[2] = len(new_sample) + (patch_on_height)
                        new_sample.append(sample[:, :, h * latent_patch_size : (h + 1) * latent_patch_size + 2, w * latent_patch_size : (w + 1) * latent_patch_size + 2])
                        patch_map.append(len(latent_offset)-1)
                        padding_idx.append(paddings)
                        # print(input_indices)
                        # print(resolution)
                        indices.append(input_indices[str(resolution)][index] + f"-{h}-{w}")
                index += 1
            if res_sample.shape[0] != 0:
                resolution_offset.append(len(latent_offset)-1)
        # padding_idx = [left_idx, top_idx, right_idx, bottom_idx]
        padding_idx = {
            "cuda": torch.cat(padding_idx, dim=0).cuda(),
            "cpu": padding_idx
        }
        latent_offset = {
            "cuda": torch.tensor(latent_offset, device="cuda", dtype=torch.int32),
            "cpu": latent_offset
        }
        resolution_offset = {
            "cuda": torch.tensor(resolution_offset, device="cuda", dtype=torch.int32),
            "cpu": resolution_offset
        }
        patch_map = {
            "cuda": torch.tensor(patch_map, device="cuda", dtype=torch.int32),
            "cpu": patch_map
        }
        return indices, padding_idx, latent_offset, resolution_offset, torch.cat(new_sample, dim=0), patch_map
        # return torch.cat(padding_idx, dim=0).cuda(), torch.tensor(latent_offset, device="cuda", dtype=torch.int32), torch.tensor(resolution_offset, device="cuda", dtype=torch.int32), torch.cat(new_sample, dim=0), torch.tensor(patch_map, device="cuda", dtype=torch.int32)

    def concat_sample(self, patch_size, new_sample, latent_offset):
        samples = dict()
        for index in range(len(latent_offset) - 1):
            patches_per_image = latent_offset[index + 1] - latent_offset[index]
            patch_on_height = int(math.sqrt(patches_per_image))
            image_size = patch_on_height * patch_size
            image_arr = list()
            for h in range(patch_on_height):
                # new_sample[latent_offset[index] + h * patch_on_height : latent_offset[index] + (h + 1) * patch_on_height]
                image_arr.append(torch.cat([new_sample[x].unsqueeze(0) for x in range(latent_offset[index] + h * patch_on_height, latent_offset[index] + (h + 1) * patch_on_height)], dim=-1))
            if str(image_size) not in samples:
                samples[str(image_size)] = list()
            samples[str(image_size)].append(torch.cat(image_arr, dim=-2))
        for key in samples:
            samples[key] = torch.cat(samples[key], dim=0)
        return samples


    def forward(
        self,
        sample,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict] = None,
        added_cond_kwargs: Optional[dict] = None,
        down_block_additional_residuals: Optional[tuple] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[tuple] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        record: bool = False,
        patch_size: int = None,
        is_sliced:bool = False,
        save_index: int = 0,
        input_indices: dict = None,
    ):
        # b, c, h, w = sample.shape
        # if patch_size is None:
        #     patch_size = h
        assert (
            class_labels is None
            and timestep_cond is None
            and attention_mask is None
            and cross_attention_kwargs is None
            and down_block_additional_residuals is None
            and mid_block_additional_residual is None
            and down_intrablock_additional_residuals is None
            and encoder_attention_mask is None
        )
        sample_key = None
        # torch.cuda.synchronize()
        start = time.time()
        if is_sliced:
            # patch_size = find_greatest_common_divisor(sample)
            indices, padding_idx, latent_offset, resolution_offset, sample, patch_map = self.split_sample(sample, patch_size, input_indices)
            encode_latens = list()
            text_embs_list = list()
            text_ids_list = list()
            latent_timesteps = list()
            for index in range(len(latent_offset["cpu"]) - 1):
                for i in range(latent_offset["cpu"][index + 1] - latent_offset["cpu"][index]):
                    encode_latens.append(encoder_hidden_states[index].unsqueeze(0))
                    latent_timesteps.append(timestep[index])
                    if added_cond_kwargs is not None and added_cond_kwargs['text_embeds'] is not None:
                        text_embs_list.append(added_cond_kwargs["text_embeds"][index].unsqueeze(0))
                        text_ids_list.append(added_cond_kwargs["time_ids"][index].unsqueeze(0))
            timestep = torch.stack(latent_timesteps, dim=0)
            encoder_hidden_states = torch.cat(encode_latens, dim=0)
            if added_cond_kwargs is not None and added_cond_kwargs['text_embeds'] is not None:
                added_cond_kwargs["text_embeds"]= torch.cat(text_embs_list, dim=0)
                added_cond_kwargs["time_ids"] = torch.cat(text_ids_list, dim=0)           
        else:
            indices = None
            self.model.conv_in.module.padding = (1,1)
            padding_idx = {"cpu": None, "cuda": None}
            latent_offset = {"cpu": None, "cuda": None}
            patch_map = {"cpu": None, "cuda": None}
            resolution_offset = {"cpu": None, "cuda": None}
            for key in sample:
                patch_size = sample[key].shape[-1]
                sample_key = key
                sample = sample[key]
                break
        # sample = sample.cuda()
        batch, _, _, _, = sample.shape
        # print(timestep)
        # normalized_batch = F.normalize(sample.view(batch, -1), p=2, dim=1)
        # print(torch.mm(normalized_batch, normalized_batch.t()))
        # if self.diff is None:
        #     self.diff = sample[12] - sample[10]
        # else:
        #     if timestep.item() > 880:
        #         # sample[3] = sample[0]
        #         # sample[7] = sample[5]
        #         sample[12] = sample[10]
        #         # sample[16] = sample[14]
        end = time.time()
        self.total_time += (end - start)
        default_overall_up_factor = 2**self.model.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.model.get_time_embed(sample=sample, timestep=timestep)
        # import pdb
        # pdb.set_trace()
        emb = self.model.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.model.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.model.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.model.time_embed_act is not None:
            emb = self.model.time_embed_act(emb)

        encoder_hidden_states = self.model.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.model.conv_in(sample, 
                                    is_sliced=is_sliced, padding_idx=padding_idx,
                                    )

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.model.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        total_blocks = 0
        down_block_res_samples = (sample,)
        for downsample_block in self.model.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    is_sliced=is_sliced,    
                    padding_idx=padding_idx,
                    latent_offset=latent_offset,
                    patch_map=patch_map,
                    resolution_offset = resolution_offset,
                    index=save_index,
                    timestep=timestep,
                    total_blocks=total_blocks,
                    input_indices=indices,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, input_indices=indices,
                                                       patch_map=patch_map,latent_offset=latent_offset,
                                                       is_sliced=is_sliced, padding_idx=padding_idx,
                                                       index=save_index, total_blocks=total_blocks, timestep=timestep,
                                                       )
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)
            total_blocks += 1
            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.model.mid_block is not None:
            if hasattr(self.model.mid_block, "has_cross_attention") and self.model.mid_block.has_cross_attention:
                sample = self.model.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    is_sliced=is_sliced,
                    padding_idx=padding_idx,
                    patch_map=patch_map,
                    latent_offset=latent_offset,
                    resolution_offset=resolution_offset,
                    index=save_index,
                    timestep=timestep,
                    total_blocks=total_blocks,
                    input_indices=indices,
                )
            else:
                sample = self.model.mid_block(sample, emb, input_indices=indices,
                                              patch_map=patch_map,latent_offset=latent_offset,
                                              is_sliced=is_sliced, padding_idx=padding_idx,
                                              index=save_index,timestep=timestep,
                    total_blocks=total_blocks,
                                              )
            total_blocks += 1
            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.model.up_blocks):
            is_final_block = i == len(self.model.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.module.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.module.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    is_sliced=is_sliced,
                    padding_idx=padding_idx,
                    patch_map=patch_map,
                    latent_offset=latent_offset,
                    resolution_offset=resolution_offset,
                    index=save_index,
                    timestep=timestep,
                    total_blocks=total_blocks,
                    input_indices=indices,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    input_indices=indices,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    patch_map=patch_map,
                    latent_offset=latent_offset,
                    index=save_index,
                    timestep=timestep,
                    total_blocks=total_blocks,
                    is_sliced=is_sliced, padding_idx=padding_idx
                )
            total_blocks += 1

        # print(total_blocks)

        # 6. post-process
        if self.model.conv_norm_out:
            # sample = self.model.conv_norm_out(sample, 
            #                                   is_sliced=is_sliced, latent_offset=latent_offset
            #                                   )
            # sample = self.model.conv_act(sample)
            sample = self.model.conv_norm_out(sample, is_sliced=is_sliced, latent_offset=latent_offset["cuda"], patch_map=patch_map["cuda"], padding_idx=padding_idx["cuda"])
            sample = self.model.conv_act(sample)
        sample = self.model.conv_out(sample, 
                                     is_sliced=is_sliced, padding_idx=padding_idx, is_padding=False
                                     )
        # torch.cuda.synchronize()
        start = time.time()
        # sample = sample.cpu()
        if is_sliced:
            sample = (self.concat_sample(patch_size, sample, latent_offset["cpu"]), )
        else:
            sample = ({
                sample_key: sample
            }, )
        # sample = sample.cuda()
        end = time.time()
        self.total_time += (end - start)
        return sample

    
    @property
    def add_embedding(self):
        return self.model.add_embedding
    
