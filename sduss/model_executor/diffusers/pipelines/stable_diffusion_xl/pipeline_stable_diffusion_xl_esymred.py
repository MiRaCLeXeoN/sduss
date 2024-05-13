from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
import torch
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, Type, TYPE_CHECKING

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline as DiffusersStableDiffusionXLPipeline,
    retrieve_timesteps, rescale_noise_cfg)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from ..pipeline_utils import BasePipeline
from ...image_processor import PipelineImageInput
from .pipeline_stable_diffusion_xl_esymred_utils import (
    StableDiffusionXLEsymredPipelinePrepareInput, StableDiffusionXLEsymredPipelinePrepareOutput,
    StableDiffusionXLEsymredPipelineStepInput, StableDiffusionXLEsymredPipelineStepOutput,
    StableDiffusionXLEsymredPipelinePostInput, StableDiffusionXLEsymredPipelineOutput,
    StableDiffusionXLEsymredPipelineSamplingParams)

from sduss.model_executor.modules.resnet import SplitModule
from sduss.worker import WorkerRequest

if TYPE_CHECKING:
    from sduss.worker import WorkerRequestDictType

class ESyMReDStableDiffusionXLPipeline(DiffusersStableDiffusionXLPipeline, BasePipeline):
    SUPPORT_MIXED_PRECISION = True

    @classmethod
    def instantiate_pipeline(cls, **kwargs):
        sub_modules: Dict = kwargs.pop("sub_modules", {})

        unet = sub_modules.pop("unet", None)

        # Lazy import to avoid cuda extension building.
        from sduss.model_executor.modules.unet import PatchUNet
        unet = PatchUNet(unet)
        sub_modules["unet"] = unet

        return cls(**sub_modules)
    
    @staticmethod
    def get_sampling_params_cls() -> Type[StableDiffusionXLEsymredPipelineSamplingParams]:
        return StableDiffusionXLEsymredPipelineSamplingParams

    
    def __post_init__(self):
        self.needs_upcasting = (self.vae.dtype == torch.float16 and self.vae.config.force_upcast)
        if self.needs_upcasting:
            self.upcast_vae()
            self.vae_upcast_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype

        
    @torch.inference_mode()
    def prepare_inference(
        self,
        worker_reqs: "WorkerRequestDictType" = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
    ) -> None:
        resolution_list = list(worker_reqs.keys())
        resolution_list.sort()
    
        # 0. Collect args
        worker_reqs_list: List[WorkerRequest] = []
        prompt = []
        prompt_2 = []
        negative_prompt = []
        negative_prompt_2 = []
        num_steps_list = []

        for res in resolution_list:
            for req in worker_reqs[res]:
                worker_reqs_list.append(req)
                prompt.append(req.sampling_params.prompt)
                prompt_2.append(req.sampling_params.prompt_2)
                negative_prompt.append(req.sampling_params.negative_prompt)
                negative_prompt_2.append(req.sampling_params.negative_prompt_2)
                num_steps_list.append(req.sampling_params.num_inference_steps)
        

        # original_size = original_size or (height, width)
        # target_size = target_size or (height, width)

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self.scheduler.batch_set_timesteps(worker_reqs_list, device=device)

        # 5. Prepare latent variables
        latent_list = []
        num_channels_latents = self.unet.config.in_channels
        for res in resolution_list:
            for req in worker_reqs[res]:
                if req.sampling_params.latents is None:
                    latent = self.prepare_latents(
                        1,
                        num_channels_latents,
                        req.sampling_params.height,
                        req.sampling_params.width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        None)
                else:
                    base_shape = (1, num_channels_latents, req.sampling_params.height // self.vae_scale_factor,
                                  req.sampling_params.width // self.vae_scale_factor)
                    latent = req.sampling_params.latents.reshape(base_shape).to(device)
                latent_list.append(latent)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        # ! Here we use (1024, 1024) for all resolutions
        add_time_ids = self._get_add_time_ids(
            (1024, 1024),
            crops_coords_top_left,
            (1024, 1024),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids
        
        # * if do_classifier_free, tensors will be cat before each denoising steps

        # add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)  # ! Propagate (1024, 1024)
        negative_add_time_ids = negative_add_time_ids.to(device)

        # * We won't step inside
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop

        # 8.1 Apply denoising_end
        # ! We won't step inside
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        # ! We won't step inside
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latent_list[0].dtype)
        
        for i, req in enumerate(worker_reqs_list):
            req.sampling_params.prompt_embeds = prompt_embeds[i].unsqueeze(0)
            req.sampling_params.negative_prompt_embeds = negative_prompt_embeds[i].unsqueeze(0)
            req.sampling_params.latents = latent_list[i]
            # Create prepare output
            req.prepare_output = StableDiffusionXLEsymredPipelinePrepareOutput(
                pooled_prompt_embeds=pooled_prompt_embeds[i].unsqueeze(0),  # (1, 1280)
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[i].unsqueeze(0),
                add_time_ids=add_time_ids,  # ! broadcst
                negative_add_time_ids=negative_add_time_ids,  # ! broadcast
                timestep_cond=timestep_cond,
                extra_step_kwargs=extra_step_kwargs,
                device=device,
                do_classifier_free_guidance=self.do_classifier_free_guidance)


    @torch.inference_mode()
    def denoising_step(
        self,
        worker_reqs: Dict[str, List[WorkerRequest]],
        do_classifier_free_guidance: bool = True,
        guidance_rescale: float = 0.0,
        guidance_scale: float = 5.0,
        timestep_cond: torch.Tensor = None,
        extra_step_kwargs: Dict = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        is_sliced: bool = False,
        patch_size: int = 256,   
    ) -> None:
        # keep the iteration in fixed order
        resolution_list = list(worker_reqs.keys())
        resolution_list.sort(key= lambda res_str: int(res_str))
        # Collect args 
        latent_dict: Dict[str, torch.Tensor] = {}
        prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        negative_prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        pooled_prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        negative_pooled_prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        timestep_dict: Dict[str, torch.Tensor] = {}
        add_time_ids_dict: Dict[str, torch.Tensor] = {}

        for res in resolution_list:
            local_latent_list = []
            local_prompt_embeds_list = []
            local_negative_prompt_embeds_list = []
            local_pooled_prompt_embeds_list = []
            local_negative_pooled_prompt_embeds_list = []
            local_timestep_list = []
            local_add_time_ids_list = []
            for req in worker_reqs[res]:
                local_latent_list.append(req.sampling_params.latents)
                local_prompt_embeds_list.append(req.sampling_params.prompt_embeds)
                local_negative_prompt_embeds_list.append(req.sampling_params.negative_prompt_embeds)
                local_pooled_prompt_embeds_list.append(req.prepare_output.pooled_prompt_embeds)
                local_negative_pooled_prompt_embeds_list.append(req.prepare_output.negative_pooled_prompt_embeds)
                local_timestep_list.append(req.scheduler_states.get_next_timestep())
                if do_classifier_free_guidance:
                    # add_time_ids has different memory layout. It must be collected here
                    local_add_time_ids_list.append(req.prepare_output.negative_add_time_ids)
                local_add_time_ids_list.append(req.prepare_output.add_time_ids)
            latent_dict[res] = torch.cat(local_latent_list, dim=0)
            prompt_embeds_dict[res] = torch.cat(local_prompt_embeds_list, dim=0)
            negative_prompt_embeds_dict[res] = torch.cat(local_negative_prompt_embeds_list, dim=0)
            pooled_prompt_embeds_dict[res] = torch.cat(local_pooled_prompt_embeds_list, dim=0)
            negative_pooled_prompt_embeds_dict[res] = torch.cat(local_negative_pooled_prompt_embeds_list, dim=0)
            timestep_dict[res] = torch.tensor(data=local_timestep_list, dtype=req.scheduler_states.timesteps.dtype,
                                              device=req.scheduler_states.timesteps.device)
            add_time_ids_dict[res] = torch.cat(local_add_time_ids_list, dim=0)
        
        # We shoule preserve the lantent_dict as original
        latent_input_dict : Dict[str, torch.Tensor] = {}

        # classifier free
        if do_classifier_free_guidance:
            prompt_embeds_list: List[torch.Tensor] = []
            timestep_list: List[torch.Tensor] = []
            add_text_embeds_list: List[torch.Tensor] = []
            add_time_ids_list: List[torch.Tensor] = []
            for res in resolution_list:
                latent_input_dict[res] = torch.cat([latent_dict[res]] * 2, dim=0)
                prompt_embeds_list.append(negative_prompt_embeds_dict[res])
                prompt_embeds_list.append(prompt_embeds_dict[res])
                add_text_embeds_list.append(negative_pooled_prompt_embeds_dict[res])
                add_text_embeds_list.append(pooled_prompt_embeds_dict[res])
                timestep_list.extend([timestep_dict[res]] * 2)
                add_time_ids_list.append(add_time_ids_dict[res])  # neg has been integrated already
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
            add_text_embeds = torch.cat(add_text_embeds_list, dim=0)
            t = torch.cat(timestep_list, dim=0)
            add_time_ids = torch.cat(add_time_ids_list, dim=0)
        else:
            prompt_embeds_list = []
            timestep_list = []
            add_text_embeds_list: List[torch.Tensor] = []
            add_time_ids_list: List[torch.Tensor] = []
            for res in resolution_list:
                latent_input_dict[res] = latent_dict[res]
                prompt_embeds_list.append(prompt_embeds_dict[res])
                add_text_embeds_list.append(pooled_prompt_embeds_dict[res])
                timestep_list.append(timestep_dict[res])
                add_time_ids_list.append(add_time_ids_dict[res])
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
            add_text_embeds = torch.cat(add_text_embeds_list, dim=0)
            t = torch.cat(timestep_list, dim=0)
            add_time_ids = torch.cat(add_time_ids_list, dim=0)
        
        # Scale input
        for res in resolution_list:
            latent_input_dict[res] = self.scheduler.batch_scale_model_input(worker_reqs=worker_reqs[res],
                                                                      samples=latent_input_dict[res],
                                                                      timestep_list=timestep_dict[res])
        
        # predict the noise residual
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        # if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        #     added_cond_kwargs["image_embeds"] = image_embeds

        # TODO: Clean up
        for res in latent_input_dict:
            print(f"latent[{res}].shape={latent_input_dict[res].shape}")
        print(f"{is_sliced=}, {patch_size=}")
        # print(f"{t.shape=}, {t=}")
        # print(f"{prompt_embeds.shape=}, {prompt_embeds=}")
        # for name in added_cond_kwargs:
        #     print(f"added_cond_kwargs[{name}].shape={added_cond_kwargs[name].shape}, {added_cond_kwargs[name]}")

        noise_pred = self.unet(
            latent_input_dict,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            is_sliced=is_sliced,
            patch_size=patch_size,
        )[0]

        # for res in noise_pred:
        #     print(f"noise_pred[{res}].shape={noise_pred[res].shape}, {noise_pred[res]}")

        # perform guidance
        for res, res_split_noise in noise_pred.items():
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = res_split_noise.chunk(2)
                res_split_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # latents[resolution] = self.scheduler.step(res_split_noise, t, latents[resolution], **extra_step_kwargs).prev_sample

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                res_split_noise = rescale_noise_cfg(res_split_noise, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            # latents[resolution] = schedulers[resolution].step(res_split_noise, t, latents[resolution], **extra_step_kwargs, return_dict=False)[0]
            latent_dict[res] = self.scheduler.batch_step(worker_reqs[res], res_split_noise, timestep_dict[res], 
                                                         latent_dict[res], **extra_step_kwargs, return_dict=False)
        # Callbacks on step end have been removed

        for res in resolution_list:
            for i, req in enumerate(worker_reqs[res]):
                req.scheduler_states.update_states_one_step()
                req.sampling_params.latents = latent_dict[res][i].unsqueeze(dim=0)


    @torch.inference_mode()
    def post_inference(
        self,
        worker_reqs: "WorkerRequestDictType",
        output_type: str = "pil",
    ) -> None:
        latent_dict: Dict[str, torch.Tensor] = {}
        for res in worker_reqs:
            latent_list = []
            for req in worker_reqs[res]:
                latent_list.append(req.sampling_params.latents)
            latent_dict[res] = torch.cat(latent_list, dim=0)
        device = latent_dict[res].device
        dtype = latent_dict[res].dtype

        images = {}
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            # upcast has been moved to __post_init__
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latent_dict[res].device, latent_dict[res].dtype)
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latent_dict[res].device, latent_dict[res].dtype)

            for res in latent_dict:
                if self.needs_upcasting:
                    latent_dict[res] = latent_dict[res].to(self.vae_upcast_dtype)

                # unscale/denormalize the latents
                # denormalize with the mean and std if available and not None
                if has_latents_mean and has_latents_std:
                    latent_dict[res] = latent_dict[res] * latents_std / self.vae.config.scaling_factor + latents_mean
                else:
                    latent_dict[res] = latent_dict[res] / self.vae.config.scaling_factor

                image = self.vae.decode(latent_dict[res], return_dict=False)[0]
                images[res] = image

                # ! We don't cast back
                # if needs_upcasting:
                #     self.vae.to(dtype=torch.float16)
        else:
            images = latent_dict

        if not output_type == "latent":
            for res in images:
            # apply watermark if available
                # if self.watermark is not None:
                #     images[resolution] = self.watermark.apply_watermark(images[resolution])
                image = self.image_processor.postprocess(images[res], output_type=output_type)

                for i, req in enumerate(worker_reqs[res]):
                    req.output = StableDiffusionXLEsymredPipelineOutput(
                        images=image[i],
                        nsfw_content_detected=None,
                    )
        else:
            raise NotImplementedError


