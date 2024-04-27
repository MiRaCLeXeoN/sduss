import inspect
from typing import Optional, List, Tuple, Union, Dict, Any, Callable

import torch

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps, rescale_noise_cfg)

from ..pipeline_utils import BasePipeline
from ...image_processor import PipelineImageInput
from sduss.model_executor.modules.unet import PatchUNet
from sduss.model_executor.modules.resnet import SplitModule
from sduss.worker import WorkerRequest

class ESyMReDStableDiffusionPipeline(BasePipeline):
    def __init__(self, pipeline: StableDiffusionPipeline):
        self.pipeline = pipeline

    @classmethod
    def instantiate_pipeline(cls, **kwargs):
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5")
        sub_modules: Dict = kwargs.pop("sub_modules", {})
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
    
    @torch.inference_mode()
    def prepare_inference(
        self,
        worker_reqs: List[WorkerRequest] = None,
        prompt: List[str] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs
    ) -> None:
        pass


    
    @torch.inference_mode()
    def denoising_step(
        self,
        worker_reqs: List[WorkerRequest],
        timestep_cond: torch.Tensor,
        added_cond_kwargs: Optional[Dict],
        extra_step_kwargs: Dict,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]],
        callback_on_step_end_tensor_inputs: List[str],
        do_classifier_free_guidance: bool,
        guidance_rescale: float,
        guidance_scale: float,
        cross_attention_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        pass
        
    
    
    @torch.inference_mode()
    def post_inference(
        self,
        worker_reqs: List[WorkerRequest],
        output_type: str,
        device: torch.device,
        prompt_embeds_dtype: torch.dtype,
        generator: torch.Generator,
    ) -> None:
        pass
    


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
        self.scheduler = dict()
        for resolution in latents:
            self.scheduler[resolution] = PNDMScheduler(num_train_timesteps=1000,
                                                        beta_start=0.00085,
                                                        beta_end=0.012,
                                                        skip_prk_steps=True,
                                                        steps_offset=1,
                                                        beta_schedule="scaled_linear")
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
        for key in latents:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler[key], num_inference_steps_total, device, timesteps_total)
            latents[key] = latents[key] * self.scheduler[key].init_noise_sigma
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
        torch.cuda.synchronize()
        start = time.time()
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipeline._interrupt:
                    continue
                latent_model_inputs = dict()
                # expand the latents if we are doing classifier free guidance
                for resolution in latents:
                    latent_model_inputs[resolution] = torch.cat([latents[resolution]] * 2) if self.pipeline.do_classifier_free_guidance else latents[resolution]
                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_inputs[resolution] = self.scheduler[resolution].scale_model_input(latent_model_inputs[resolution], t)

                noise_pred = self.pipeline.unet(
                    latent_model_inputs,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.pipeline.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    is_sliced=is_sliced,
                    patch_size=patch_size
                )[0]

                for resolution, res_split_noise in noise_pred.items():
                    if self.pipeline.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = res_split_noise.chunk(2)
                        res_split_noise = noise_pred_uncond + self.pipeline._guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.pipeline.do_classifier_free_guidance and self.pipeline._guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        res_split_noise = rescale_noise_cfg(res_split_noise, noise_pred_text, guidance_rescale=self.pipeline._guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents[resolution] = self.scheduler[resolution].step(res_split_noise, t, latents[resolution], **extra_step_kwargs, return_dict=False)[0]
                #     self.pipeline.scheduler._step_index -= 1
                # self.pipeline.scheduler._step_index += 1
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()
            # make sure the VAE is in float32 mode, as it overflows in float16
        
        torch.cuda.synchronize()
        end = time.time()
        total_unet_time = end - start
        start =time.time()
        images = list()
        for resolution, res_split_latents in latents.items():
            if res_split_latents.shape[0] != 0:
                image = self.pipeline.vae.decode(res_split_latents / self.pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]
                # print(self.pipeline.vae.decode(res_split_latents, return_dict=False)[0])
                images.append(self.pipeline.image_processor.postprocess(image, output_type="pil"))
        torch.cuda.synchronize()
        end = time.time()
        total_post_process_time = end - start
        return images, total_unet_time, total_post_process_time
