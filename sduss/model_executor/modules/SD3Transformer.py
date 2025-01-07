from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_module import BaseModule, BaseModel
from diffusers.models.transformers.transformer_2d import Transformer2DModel, Transformer2DModelOutput
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models import SD3Transformer2DModel
from .transformer import PatchJointTransformerBlock
from .attention import PatchSD3Attention
from .utils import split_sample_sd3, concat_sample
from .cache_manager import CacheManager
# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.attention import JointTransformerBlock

class PatchSD3Transformer2DModel(BaseModel):
    def __init__(
        self,
        model: SD3Transformer2DModel,
    ):
        for name, module in model.named_modules():
            if isinstance(module, BaseModule):
                continue
            
            for subname, submodule in module.named_children():
                if isinstance(submodule, Attention):
                    wrapped_submodule = PatchSD3Attention(submodule)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, JointTransformerBlock):
                    wrapped_submodule = PatchJointTransformerBlock(submodule)
                    setattr(module, subname, wrapped_submodule)
                
        # self.diff = None
        super(PatchSD3Transformer2DModel, self).__init__(model)
        self.state_input = []
        self.encoder_input = []
        self.state_output = []
        self.encoder_output = []
        self.hidden_state_cache = [] 
        self.encoder_state_cache = []
        self.hidden_state_index = [] 
        self.encoder_state_index = []
        self.previous_mask = []
        for index_block, block in enumerate(self.model.transformer_blocks):
            self.state_input.append(CacheManager())
            self.encoder_input.append(CacheManager())
            self.state_output.append(CacheManager())
            self.encoder_output.append(CacheManager())
            # self.previous_mask.append(None)

    def forward(
        self,
        hidden_states: dict,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        patch_size: int = None,
        is_sliced:bool = False,
        save_index: int = 0,
        input_indices: dict = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        temb = self.model.time_text_embed(timestep, pooled_projections)
        for resolution in hidden_states:
            hidden_states[resolution] = self.model.pos_embed(hidden_states[resolution])  # takes care of adding positional embeddings too.
        if is_sliced:
            # patch_size = find_greatest_common_divisor(sample)
            indices, encoder_indices, latent_offset, resolution_offset, hidden_states = split_sample_sd3(hidden_states, patch_size, input_indices)
            # encode_latens = list()
            latent_timesteps = list()
            # pooled_projections_list = list()
            temb_latent_list = list()
            for index in range(len(latent_offset["cpu"]) - 1):
                for i in range(latent_offset["cpu"][index + 1] - latent_offset["cpu"][index]):
                    # encode_latens.append(encoder_hidden_states[index].unsqueeze(0))
                    latent_timesteps.append(timestep[index])
                    # pooled_projections_list.append(pooled_projections[index])
                    temb_latent_list.append(temb[index])
            state_timestep = torch.stack(latent_timesteps, dim=0)
            # encoder_hidden_states = torch.cat(encode_latens, dim=0)
            # pooled_projections = torch.stack(pooled_projections_list, dim=0)
            temb_latent = torch.stack(temb_latent_list, dim=0)
        else:
            indices = None
            state_timestep = None
            encoder_indices = None
            latent_offset = {"cpu": None, "cuda": None}
            resolution_offset = {"cpu": None, "cuda": None}
            temb_latent = temb.clone()
            for key in hidden_states:
                patch_size = hidden_states[key].shape[-1]
                sample_key = key
                hidden_states = hidden_states[key]
                break
        height, width = hidden_states.shape[-2:]
        # temb_latent = self.model.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.model.context_embedder(encoder_hidden_states)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.model.image_proj(ip_adapter_image_embeds, timestep)

            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)
        # mse_loss = nn.MSELoss(reduction='none')
        # if not os.path.exists(f"cache/{save_index}/{timestep[0]}"):
        #     os.makedirs(f"cache/{save_index}/{timestep[0]}")
        for index_block, block in enumerate(self.model.transformer_blocks):
            # Skip specified layers
            is_skip = True if skip_layers is not None and index_block in skip_layers else False

            if torch.is_grad_enabled() and self.model.gradient_checkpointing and not is_skip:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )
            elif not is_skip:
                
                state_mask = self.state_input[index_block].get_sd3_mask(indices, hidden_states, index_block, state_timestep)
                # encoder_mask = self.encoder_input[index_block].get_sd3_mask(encoder_indices, encoder_hidden_states, index_block, timestep, False)
                # if index_block >= 1:
                #     state_mask = state_mask | self.previous_mask[index_block - 1]
                # self.previous_mask[index_block] = ~state_mask
                # encoder_mask = self.encoder_input.get_sd3_mask(indices, encoder_hidden_states, index_block, timestep, False)
                # print(timestep[0])
                '''
                if timestep[0] > 995:
                    self.hidden_state_index[index_block] = hidden_states
                    
                    self.encoder_state_index[index_block] = encoder_hidden_states
                else:
                    hidden_states_loss = mse_loss(self.hidden_state_index[index_block].to(torch.float32), hidden_states.to(torch.float32)).mean(dim=(-1,-2))
                    torch.save(hidden_states_loss, f"cache/{save_index}/{timestep[0]}/{index_block}-input-state.pt")
                    if encoder_hidden_states is not None:
                        encoder_state_loss = mse_loss(self.encoder_state_index[index_block].to(torch.float32), encoder_hidden_states.to(torch.float32)).mean(dim=(-1,-2))
                        torch.save(encoder_state_loss, f"cache/{save_index}/{timestep[0]}/{index_block}-input-encoder.pt")
                '''
                # print(state_mask.sum() / len(state_mask))
                if state_mask.sum() > 0:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        temb_latent=temb_latent,
                        joint_attention_kwargs=joint_attention_kwargs,
                        is_sliced=is_sliced,
                        latent_offset=latent_offset,
                        resolution_offset=resolution_offset,
                        input_indices=indices,
                        encoder_indices=encoder_indices,
                        state_mask=state_mask,
                    )
                    '''
                    if timestep[0] > 995:
                        self.hidden_state_cache[index_block] = hidden_states
                        if encoder_hidden_states is not None:
                            self.encoder_state_cache[index_block] = encoder_hidden_states
                        
                    else:
                        
                        # print(torch.isnan(hidden_states).any())
                        # print(torch.isinf(hidden_states).any())
                        
                        loss = mse_loss(self.hidden_state_cache[index_block].to(torch.float32), hidden_states.to(torch.float32)).mean(dim=(-1,-2))
                        mask = loss < 0.1
                        torch.save(loss, f"cache/{save_index}/{timestep[0]}/{index_block}-output-state.pt")
                        if self.previous_mask[index_block] is not None:
                            mask = mask & self.previous_mask[index_block]
                        # if index_block >= 1 and self.previous_mask[index_block - 1] is not None:
                        #         mask = mask & self.previous_mask[index_block - 1]
                        self.previous_mask[index_block] = ~mask
                        hidden_states[mask] = self.hidden_state_cache[index_block][mask].to(torch.float16)
                        # print(mask.sum())
                        self.hidden_state_cache[index_block] = hidden_states
                        if encoder_hidden_states is not None:
                            loss = mse_loss(self.encoder_state_cache[index_block].to(torch.float32), encoder_hidden_states.to(torch.float32)).mean(dim=(-1,-2))
                            # mask = loss > 0.1
                            # print(mask.sum())
                            # encoder_hidden_states[mask] = self.encoder_state_cache[index_block][mask]
                            torch.save(loss, f"cache/{save_index}/{timestep[0]}/{index_block}-output-encoder.pt")
                            self.encoder_state_cache[index_block] = encoder_hidden_states
                        # print(timestep)
                    '''
                
                        # print(mse_loss(self.encoder_state_cache, encoder_hidden_states).mean(dim=(-1)))
                if self.encoder_output[index_block] is None:
                    self.encoder_output[index_block] = encoder_hidden_states
                    # self.state_output = hidden_states
                else:
                    if state_mask.sum() == 0:
                        encoder_hidden_states = self.encoder_output[index_block]
                #         hidden_states = self.state_output
                    else:
                        self.encoder_output[index_block] = encoder_hidden_states
                #         self.state_output = hidden_states
                hidden_states = self.state_output[index_block].save_and_get_block_states(indices, hidden_states, state_mask)
                
                # if encoder_hidden_states is not None:
                #     encoder_hidden_states = self.encoder_output[index_block].save_and_get_block_states(encoder_indices, encoder_hidden_states, encoder_mask)
                # encoder_hidden_states = self.encoder_output.save_and_get_block_states(indices, encoder_hidden_states, encoder_mask)
            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.model.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        hidden_states = self.model.norm_out(hidden_states, temb_latent)
        hidden_states = self.model.proj_out(hidden_states)

        # unpatchify
        ps = self.model.config.patch_size
        
        if is_sliced:
            hidden_states = concat_sample(patch_size, hidden_states, latent_offset["cpu"])
        else:
            hidden_states = {
                sample_key: hidden_states
            }
        for resolution in hidden_states:
            height = (int(resolution) // 8) // ps
            width = (int(resolution) // 8) // ps
            hidden_states[resolution] = hidden_states[resolution].reshape(
                shape=(hidden_states[resolution].shape[0], height, width, ps, ps, self.model.out_channels)
            )
            hidden_states[resolution] = torch.einsum("nhwpqc->nchpwq", hidden_states[resolution])
            hidden_states[resolution] = hidden_states[resolution].reshape(
                shape=(hidden_states[resolution].shape[0], self.model.out_channels, height * ps, width * ps)
            )
        
        
        return (hidden_states, )
