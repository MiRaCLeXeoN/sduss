from typing import overload, Union, Optional, List, Dict, Any, Callable, Type

import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline as DiffusersStableDiffusionPipeline,
    retrieve_timesteps, rescale_noise_cfg)

from .pipeline_stable_diffusion_utils import (
    StableDiffusionPipelineOutput, StableDiffusionPipelineStepOutput,
    StableDiffusionPipelinePrepareOutput, StableDiffusionPipelineStepInput,
    StableDiffusionPipelineSamplingParams)

from ..pipeline_utils import BasePipeline
from ...image_processor import PipelineImageInput

from sduss.worker import WorkerRequest


class StableDiffusionPipeline(DiffusersStableDiffusionPipeline, BasePipeline):

    @classmethod
    def instantiate_pipeline(cls, **kwargs):
        sub_modules: Dict = kwargs.pop("sub_modules", {})
        return cls(**sub_modules)
    
    def prepare_input_dict(
        self,
        worker_reqs: List[Warning]
    ) -> Dict:
        pass

    @staticmethod
    def get_sampling_params_cls() -> Type[StableDiffusionPipelineSamplingParams]:
        return StableDiffusionPipelineSamplingParams

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
    ) -> StableDiffusionPipelinePrepareOutput:
        r"""
        Prepare denoising. Arguments will be stored inside the object. And all necessary steps before
        UNet iteration will be performed here.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
                Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
                if `do_classifier_free_guidance` is set to `True`.
                If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # callback and callback_steps are removed

        assert prompt is None and negative_prompt is None and num_inference_steps is None and latents is None, (
            "These parameters must be passed via worker_reqs.")
        assert worker_reqs is not None, "Worker requests must be passed!"
        
        # Extract params:
        prompt: List = []
        negative_prompt: List = []
        for req in worker_reqs:
            assert req.sampling_params.height
            prompt.append(req.sampling_params.prompt)
            negative_prompt.append(req.sampling_params.negative_prompt)
        
        # 0. Set height and width of Image
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
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

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            1,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # ! We do concatenation before each Unet iteration
        # if self.do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size,
                self.do_classifier_free_guidance,
            )
            
        # 4. Prepare timesteps
        # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self.scheduler.batch_set_timesteps(worker_reqs, device=device)

        # 5. Prepare latent variables
        # Latents must be kept separate, since it will be stored independently
        num_channels_latents = self.unet.config.in_channels
        latents: List = []
        base_shape = (1, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        for req in worker_reqs:
            if req.sampling_params.latents is None:
                latent = self.prepare_latents(
                    batch_size=1,
                    num_channels_latents=0,
                    height=height,
                    width=width,
                    dtype=prompt_embeds.dtype,
                    device=device)
            else:
                latent = req.sampling_params.latents.reshape(base_shape)
            latents.append(latent)
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        # This will not be reached
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # self._num_timesteps = len(timesteps)

        # Store prepare result to requests
        for i, req in enumerate(worker_reqs):
            # update necessary variables
            req.sampling_params.latents = latents[i]
            req.sampling_params.prompt_embeds = prompt_embeds[i]
            req.sampling_params.negative_prompt_embeds = negative_prompt_embeds[i]
            # Create prepare output
            prepare_output = StableDiffusionPipelinePrepareOutput(
                timestep_cond=timestep_cond,
                added_cond_kwargs=added_cond_kwargs,
                extra_step_kwargs=extra_step_kwargs,
                device=device,
                do_classifier_free_guidance=self.do_classifier_free_guidance,)
            req.prepare_output = prepare_output


    def denoise_step(
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
        # Prepare inputs
        latents: List[torch.Tensor] = []
        prompt_embeds: List[torch.Tensor] = []
        negative_prompt_embeds: List[torch.Tensor] = []
        timesteps: List[Union[float, int]] = [] 
        timestep_idxs: List[int] = []
        for req in worker_reqs:
            latents.append(req.sampling_params.latents)
            prompt_embeds.append(req.sampling_params.prompt_embeds.unsqueeze())
            negative_prompt_embeds.append(req.sampling_params.negative_prompt_embeds.unsqueeze())
            timesteps.append(req.scheduler_states.get_next_timestep())
            timestep_idxs.append(req.scheduler_states.get_step_idx())
            # TODO(MX): Check tensor shape here
        
        latents = torch.cat(latents, dim=0)
        prompt_embeds = torch.cat(prompt_embeds, dim=0)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=0)
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            timesteps = timesteps * 2
        t = torch.tensor(data=timesteps, dtype=worker_reqs[0].scheduler_states.timesteps.dtype)

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.batch_scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if do_classifier_free_guidance and guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1
        # * We don't need to split latents here, since it is not updated by the previous steps.
        latents = self.scheduler.batch_step(worker_reqs, noise_pred, t, **extra_step_kwargs, return_dict=False)

        # ! This will not be reached.
        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            # ! If we want this work, callback_on_step_end should be batch-compatible
            callback_outputs = callback_on_step_end(self, timestep_idxs, t, callback_kwargs)

            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
            negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
        
        # Update parameters
        for i, req in enumerate(worker_reqs):
            req.scheduler_states.update_states_one_step()
            req.sampling_params.latents = latents[i]
            # req.step_output = StableDiffusionPipelineStepOutput()  # Nothing to store
                
    
    def post_inference(
        self,
        worker_reqs: List[WorkerRequest],
        output_type: str,
        device: torch.device,
        prompt_embeds_dtype: torch.dtype,
        generator: torch.Generator,
        return_dict: bool,
    ) -> StableDiffusionPipelineOutput:
        latents: List[torch.Tensor] = []
        for req in worker_reqs:
            latents.append(req.sampling_params.latents.unsqueeze())
        latents = torch.cat(latents, dim=0)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds_dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        # TODO(MX): We do not need cpu offload function
        self.maybe_free_model_hooks()

        for i, req in enumerate(worker_reqs):
            # TODO(MX): nsfw_content_detected need to be processed for each request.
            req.output = StableDiffusionPipelineOutput(
                images=image[i],
                nsfw_content_detected=None,
            )


    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

