import inspect
from typing import Optional, List, Tuple, Union, Dict

import torch

from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from sduss.model_executor.diffusers import BasePipeline
from sduss.model_executor.diffusers.image_processor import PipelineImageInput
from sduss.model_executor.modules.unet import PatchUNet
from sduss.model_executor.modules.resnet import SplitModule

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class ESyMReDStableDiffusionPipeline(BasePipeline):
    def __init__(self, pipeline: StableDiffusionPipeline):
        self.pipeline = pipeline

    @classmethod
    def instantiate_pipeline(cls, **kwargs):
        sub_modules: Dict = kwargs.pop("sub_modules", {})
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5")
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)

        unet = sub_modules.pop("unet", None)
        if unet is None:
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet")
        unet = PatchUNet(unet)

        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **sub_modules, **kwargs)

        return cls(pipeline)


    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)


    def get_profile(self, profile_dir):
        for name, module in self.pipeline.unet.named_modules():
            for subname, submodule in module.named_children():
                if isinstance(submodule, SplitModule):
                    submodule.get_profile(profile_dir)


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        is_sliced=False,
        patch_size=1,
        **kwargs,
    ):

        # height = height or self.unet.config.sample_size * self.vae_scale_factor
        # width = width or self.unet.config.sample_size * self.vae_scale_factor

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        # do_classifier_free_guidance = guidance_scale > 1.0 and self.pipeline.unet.config.time_cond_proj_dim is None

        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._guidance_rescale = guidance_rescale
        self.pipeline._clip_skip = clip_skip
        self.pipeline._cross_attention_kwargs = cross_attention_kwargs
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
            self.pipeline._cross_attention_kwargs.get("scale", None) if self.pipeline._cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.pipeline.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.pipeline.clip_skip,
        )

        base_offset = 0
        embeds = list()
        if self.pipeline.do_classifier_free_guidance:
            for resolution in latents:
                embeds.append(torch.cat([negative_prompt_embeds[base_offset:base_offset+latents[resolution].shape[0]], prompt_embeds[base_offset:base_offset+latents[resolution].shape[0]]], dim=0))
                base_offset = base_offset+latents[resolution].shape[0]
            prompt_embeds = torch.cat(embeds, dim=0)
        prompt_embeds = prompt_embeds.to(device)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.pipeline.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.pipeline.scheduler, num_inference_steps, device, timesteps)

        for key in latents:
            latents[key] = latents[key] * self.pipeline.scheduler.init_noise_sigma
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta)

        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        timestep_cond = None
        if self.pipeline.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipeline.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipeline.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipeline._interrupt:
                    continue
                latent_model_inputs = dict()
                # expand the latents if we are doing classifier free guidance
                for resolution in latents:
                    latent_model_inputs[resolution] = torch.cat([latents[resolution]] * 2) if self.pipeline.do_classifier_free_guidance else latents[resolution]
                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_inputs[resolution] = self.pipeline.scheduler.scale_model_input(latent_model_inputs[resolution], t)

                noise_pred = self.pipeline.unet(
                    latent_model_inputs,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.pipeline.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    # is_sliced=is_sliced,
                    # patch_size=patch_size
                )[0]

                for resolution, res_split_noise in noise_pred.items():
                    if self.pipeline.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = res_split_noise.chunk(2)
                        res_split_noise = noise_pred_uncond + self.pipeline._guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.pipeline.do_classifier_free_guidance and self.pipeline._guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        res_split_noise = rescale_noise_cfg(res_split_noise, noise_pred_text, guidance_rescale=self.pipeline._guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents[resolution] = self.pipeline.scheduler.step(res_split_noise, t, latents[resolution], **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()
            # make sure the VAE is in float32 mode, as it overflows in float16
            
                            
        images = list()
        for resolution, res_split_latents in latents.items():
            image = self.pipeline.vae.decode(res_split_latents / self.pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # print(self.pipeline.vae.decode(res_split_latents, return_dict=False)[0])
            images.append(self.pipeline.image_processor.postprocess(image, output_type="pil"))

        return images
