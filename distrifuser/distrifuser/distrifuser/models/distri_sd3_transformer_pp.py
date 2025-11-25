import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from torch import distributed as dist, nn
from diffusers.models import SD3Transformer2DModel
from .base_model import BaseModel
from distrifuser.modules.pp.attn import DistriSD3AttentionPP
from distrifuser.modules.base_module import BaseModule
from ..utils import DistriConfig
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers.models.transformers.transformer_sd3 import Transformer2DModelOutput, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

class DistriTransformerPP(BaseModel):  # for Patch Parallelism
    def __init__(self, model: SD3Transformer2DModel, distri_config: DistriConfig):
        assert isinstance(model, SD3Transformer2DModel)
        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                for subname, submodule in module.named_children():
                    if isinstance(submodule, Attention):
                        wrapped_submodule = DistriSD3AttentionPP(submodule, distri_config, is_first_layer=name == "transformer_blocks.0" and subname=="attn")
                        setattr(module, subname, wrapped_submodule)


        super(DistriTransformerPP,self).__init__(model, distri_config)

    
    def _forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self.model, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                # logger.warning(
                #     "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                # )
                pass
        
        distri_config = self.distri_config

        height, width = hidden_states.shape[-2:]
        hidden_states = self.model.pos_embed(hidden_states)  # takes care of adding positional embeddings too.

        # Split
        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            height = height // distri_config.n_device_per_batch
            _, n, _ = hidden_states.shape
            slice = n // distri_config.n_device_per_batch
            idx = distri_config.split_idx()
            hidden_states = hidden_states[:, idx * slice : (idx + 1) * slice, :]

        
        temb = self.model.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.model.context_embedder(encoder_hidden_states)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.model.image_proj(ip_adapter_image_embeds, timestep)

            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        for index_block, block in enumerate(self.model.transformer_blocks):
            # Skip specified layers
            is_skip = True if skip_layers is not None and index_block in skip_layers else False

            if torch.is_grad_enabled() and self.model.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = self.model._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.model.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        hidden_states = self.model.norm_out(hidden_states, temb)
        hidden_states = self.model.proj_out(hidden_states)

        # unpatchify
        patch_size = self.model.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.model.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.model.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self.model, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def forward(
        self,
        hidden_states: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        record: bool = False,
    ):
        distri_config = self.distri_config
        b, c, h, w = hidden_states.shape

        # if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
        #     slice = h // distri_config.n_device_per_batch
        #     idx = distri_config.split_idx()
        #     hidden_states = hidden_states[:, :, idx * slice : (idx + 1) * slice, :]

        if distri_config.use_cuda_graph and not record:
            raise NotImplementedError
            static_inputs = self.static_inputs

            if distri_config.world_size > 1 and distri_config.do_classifier_free_guidance and distri_config.split_batch:
                assert b == 2
                batch_idx = distri_config.batch_idx()
                hidden_states = hidden_states[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1] if torch.is_tensor(timestep) and timestep.ndim > 0 else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
                if pooled_projections is not None:
                    pooled_projections = pooled_projections[batch_idx : batch_idx + 1]
                if joint_attention_kwargs is not None:
                    for k in joint_attention_kwargs:
                        joint_attention_kwargs[k] = joint_attention_kwargs[k][batch_idx : batch_idx + 1]
                # if added_cond_kwargs is not None:
                #     for k in added_cond_kwargs:
                #         added_cond_kwargs[k] = added_cond_kwargs[k][batch_idx : batch_idx + 1]

            assert static_inputs["hidden_states"].shape == hidden_states.shape
            static_inputs["hidden_states"].copy_(hidden_states)
            if torch.is_tensor(timestep):
                if timestep.ndim == 0:
                    for b in range(static_inputs["timestep"].shape[0]):
                        static_inputs["timestep"][b] = timestep.item()
                else:
                    assert static_inputs["timestep"].shape == timestep.shape
                    static_inputs["timestep"].copy_(timestep)
            else:
                for b in range(static_inputs["timestep"].shape[0]):
                    static_inputs["timestep"][b] = timestep
            assert static_inputs["encoder_hidden_states"].shape == encoder_hidden_states.shape
            static_inputs["encoder_hidden_states"].copy_(encoder_hidden_states)
            if pooled_projections is not None:
                static_inputs["pooled_projections"].copy_(pooled_projections)
            if joint_attention_kwargs is not None:
                for k in joint_attention_kwargs:
                    static_inputs["joint_attention_kwargs"][k].copy_(joint_attention_kwargs[k])
            # if added_cond_kwargs is not None:
            #     for k in added_cond_kwargs:
            #         assert static_inputs["added_cond_kwargs"][k].shape == added_cond_kwargs[k].shape
            #         static_inputs["added_cond_kwargs"][k].copy_(added_cond_kwargs[k])

            if self.counter <= distri_config.warmup_steps:
                graph_idx = 0
            elif self.counter == distri_config.warmup_steps + 1:
                graph_idx = 1
            else:
                graph_idx = 2

            self.cuda_graphs[graph_idx].replay()
            output = self.static_outputs[graph_idx]
        else:
            if distri_config.world_size == 1:
                output = self.model(
                    hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
            elif distri_config.do_classifier_free_guidance and distri_config.split_batch:
                raise NotImplementedError
                assert b == 2
                batch_idx = distri_config.batch_idx()
                hidden_states = hidden_states[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1] if torch.is_tensor(timestep) and timestep.ndim > 0 else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
                if pooled_projections is not None:
                    pooled_projections = pooled_projections[batch_idx : batch_idx + 1]
                if joint_attention_kwargs is not None:
                    new_joint_attention_kwargs = {}
                    for k in joint_attention_kwargs:
                        new_joint_attention_kwargs[k] = joint_attention_kwargs[k][batch_idx : batch_idx + 1]
                    joint_attention_kwargs = new_joint_attention_kwargs 
                # if added_cond_kwargs is not None:
                #     new_added_cond_kwargs = {}
                #     for k in added_cond_kwargs:
                #         new_added_cond_kwargs[k] = added_cond_kwargs[k][batch_idx : batch_idx + 1]
                #     added_cond_kwargs = new_added_cond_kwargs
                output = self.model(
                    hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
                if self.output_buffer is None:
                    self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype)
                if self.buffer_list is None:
                    self.buffer_list = [torch.empty_like(output) for _ in range(distri_config.world_size)]
                dist.all_gather(self.buffer_list, output.contiguous(), async_op=False)
                # print(hidden_states.shape, output.shape, self.output_buffer.shape, torch.cat(self.buffer_list, dim=2).shape)
                torch.cat(self.buffer_list[: distri_config.n_device_per_batch], dim=2, out=self.output_buffer[0:1])
                torch.cat(self.buffer_list[distri_config.n_device_per_batch :], dim=2, out=self.output_buffer[1:2])
                output = self.output_buffer
            else:
                # We pass full hidden_states to the model, it will be sliced inside
                # the pos_embed module
                output = self._forward(
                    hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
                if self.output_buffer is None:
                    self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype)
                if self.buffer_list is None:
                    self.buffer_list = [torch.empty_like(output) for _ in range(distri_config.world_size)]
                output = output.contiguous()
                dist.all_gather(self.buffer_list, output, async_op=False)
                torch.cat(self.buffer_list, dim=2, out=self.output_buffer)
                output = self.output_buffer
            if record:
                if self.static_inputs is None:
                    self.static_inputs = {
                        "hidden_states": hidden_states,
                        "timestep": timestep,
                        "encoder_hidden_states": encoder_hidden_states,
                        "joint_attention_kwargs": joint_attention_kwargs,
                        "pooled_projections": pooled_projections,
                    }
                self.synchronize()


        output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
