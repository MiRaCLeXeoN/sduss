import torch
from typing import Optional, List, Tuple, Union, Dict, Any, Callable

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    retrieve_timesteps, rescale_noise_cfg)

from ..pipeline_utils import BasePipeline
from ...image_processor import PipelineImageInput
from sduss.model_executor.modules.unet import PatchUNet
from sduss.model_executor.modules.resnet import SplitModule

class ESyMReDStableDiffusionXLPipeline(BasePipeline):
    def __init__(self, pipeline: StableDiffusionXLPipeline):
        self.pipeline = pipeline

    @classmethod
    def instantiate_pipeline(cls, **kwargs):
        sub_modules: Dict = kwargs.pop("sub_modules", {})
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0")
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)

        unet = sub_modules.pop("unet", None)
        if unet is None:
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet")
        unet = PatchUNet(unet)

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **sub_modules, **kwargs)

        return cls(pipeline)


    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

        
    @torch.inference_mode()
    def prepare_inference() -> None:
        pass


    @torch.inference_mode()
    def denoising_step() -> None:
        pass


    @torch.inference_mode()
    def post_inference() -> None:
        pass


    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        is_sliced: bool = False,
        patch_size: int = 512,
        **kwargs,
    ):


        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        width = width or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._guidance_rescale = guidance_rescale
        self.pipeline._clip_skip = clip_skip
        self.pipeline._cross_attention_kwargs = cross_attention_kwargs
        self.pipeline._denoising_end = denoising_end
        self.pipeline._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.pipeline.cross_attention_kwargs.get("scale", None) if self.pipeline.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.pipeline.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.pipeline.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.pipeline.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.pipeline.unet.config.in_channels

        for key in latents:
            latents[key] = latents[key] * self.pipeline.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim

        add_time_ids = self.pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.pipeline._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        base_offset = 0
        embeds = list()
        add_text_embeds_list = list()
        add_time_ids_list = list()
        if self.pipeline.do_classifier_free_guidance:
            for resolution in latents:
                embeds.append(torch.cat([negative_prompt_embeds[base_offset:base_offset+latents[resolution].shape[0]], prompt_embeds[base_offset:base_offset+latents[resolution].shape[0]]], dim=0))
                add_text_embeds_list.append(torch.cat([negative_pooled_prompt_embeds[base_offset:base_offset+latents[resolution].shape[0]], add_text_embeds[base_offset:base_offset+latents[resolution].shape[0]]], dim=0))
                add_time_ids_list.append(torch.cat([negative_add_time_ids[base_offset:base_offset+latents[resolution].shape[0]], add_time_ids[base_offset:base_offset+latents[resolution].shape[0]]], dim=0))
                base_offset = base_offset+latents[resolution].shape[0]
            prompt_embeds = torch.cat(embeds, dim=0)
            add_text_embeds = torch.cat(add_text_embeds_list, dim=0)
            add_time_ids = torch.cat(add_time_ids_list, dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.pipeline.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipeline.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.pipeline.denoising_end is not None
            and isinstance(self.pipeline.denoising_end, float)
            and self.pipeline.denoising_end > 0
            and self.pipeline.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.pipeline.scheduler.config.num_train_timesteps
                    - (self.pipeline.denoising_end * self.pipeline.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipeline.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.pipeline.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipeline.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.pipeline._num_timesteps = len(timesteps)
        # torch.cuda.synchronize()
        start = time.time()
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipeline.interrupt:
                    continue
                latent_model_inputs = dict()
                # expand the latents if we are doing classifier free guidance
                
                
                for resolution in latents:
                    latent_model_inputs[resolution] = torch.cat([latents[resolution]] * 2) if self.pipeline.do_classifier_free_guidance else latents[resolution]
                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_inputs[resolution] = self.pipeline.scheduler.scale_model_input(latent_model_inputs[resolution], t)
                    
                    # latent_model_inputs[resolution] = schedulers[resolution].scale_model_input(latent_model_inputs[resolution], t)
                
                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.pipeline.unet(
                    latent_model_inputs,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.pipeline.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    is_sliced=is_sliced,
                    patch_size=patch_size,
                )[0]

                # perform guidance
                for resolution, res_split_noise in noise_pred.items():
                    if self.pipeline.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = res_split_noise.chunk(2)
                        res_split_noise = noise_pred_uncond + self.pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    # latents[resolution] = self.pipeline.scheduler.step(res_split_noise, t, latents[resolution], **extra_step_kwargs).prev_sample

                    if self.pipeline.do_classifier_free_guidance and self.pipeline.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        res_split_noise = rescale_noise_cfg(res_split_noise, noise_pred_text, guidance_rescale=self.pipeline.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    # latents[resolution] = schedulers[resolution].step(res_split_noise, t, latents[resolution], **extra_step_kwargs, return_dict=False)[0]
                    latents[resolution] = self.pipeline.scheduler.step(res_split_noise, t, latents[resolution], **extra_step_kwargs, return_dict=False)[0]
                    self.pipeline.scheduler._step_index -= 1
                self.pipeline.scheduler._step_index += 1

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.pipeline.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        # torch.cuda.synchronize()
        end = time.time()
        total_unet_time = end - start
        start = time.time()
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast
            images = dict()
            for resolution in latents:
                if latents[resolution].shape[0] != 0:
                    if needs_upcasting:
                        self.pipeline.upcast_vae()
                        latents[resolution] = latents[resolution].to(next(iter(self.pipeline.vae.post_quant_conv.parameters())).dtype)

                    # unscale/denormalize the latents
                    # denormalize with the mean and std if available and not None
                    has_latents_mean = hasattr(self.pipeline.vae.config, "latents_mean") and self.pipeline.vae.config.latents_mean is not None
                    has_latents_std = hasattr(self.pipeline.vae.config, "latents_std") and self.pipeline.vae.config.latents_std is not None
                    if has_latents_mean and has_latents_std:
                        latents_mean = (
                            torch.tensor(self.pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                        )
                        latents_std = (
                            torch.tensor(self.pipeline.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                        )
                        latents[resolution] = latents[resolution] * latents_std / self.pipeline.vae.config.scaling_factor + latents_mean
                    else:
                        latents[resolution] = latents[resolution] / self.pipeline.vae.config.scaling_factor

                    image = self.pipeline.vae.decode(latents[resolution], return_dict=False)[0]
                    images[resolution] = image

                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.pipeline.vae.to(dtype=torch.float16)
            else:
                image = latents

        if not output_type == "latent":
            for resolution in images:
            # apply watermark if available
                if self.pipeline.watermark is not None:
                    images[resolution] = self.pipeline.watermark.apply_watermark(images[resolution])

                images[resolution] = self.pipeline.image_processor.postprocess(images[resolution], output_type=output_type)

        # Offload all models
        # self.maybe_free_model_hooks()
        # torch.cuda.synchronize()
        end = time.time()
        total_post_process_time = end - start
        if not return_dict:
            return (images, total_unet_time, total_post_process_time,)

        return images, total_unet_time, total_post_process_time
