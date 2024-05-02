import inspect
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, Type

import torch

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline as DiffusersStableDiffusionPipeline,
    retrieve_timesteps, rescale_noise_cfg)

from ..pipeline_utils import BasePipeline
from ...image_processor import PipelineImageInput
from sduss.model_executor.modules.unet import PatchUNet
from sduss.model_executor.modules.resnet import SplitModule
from sduss.worker import WorkerRequest, WorkerRequestDictType
from .pipeline_stable_diffusion_esymred_utils import (
    StableDiffusionEsymredPipelinePrepareInput, StableDiffusionEsymredPipelinePrepareOutput,
    StableDiffusionEsymredPipelineStepInput, StableDiffusionEsymredPipelineStepOutput,
    StableDiffusionEsymredPipelinePostInput, StableDiffusionEsymredPipelineOutput,
    StableDiffusionEsymredPipelineSamplingParams)

class ESyMReDStableDiffusionPipeline(DiffusersStableDiffusionPipeline, BasePipeline):
    SUPPORT_MIXED_PRECISION = True

    @classmethod
    def instantiate_pipeline(cls, **kwargs):
        sub_modules: Dict = kwargs.pop("sub_modules", {})

        unet = sub_modules.pop("unet", None)
        unet = PatchUNet(unet)
        sub_modules["unet"] = unet

        return cls(**sub_modules)
    
    @staticmethod
    def get_sampling_params_cls() -> Type[StableDiffusionEsymredPipelineSamplingParams]:
        return StableDiffusionEsymredPipelineSamplingParams
    
    
    def __post_init__(self):
        pass


    @torch.inference_mode()
    def prepare_inference(
        self,
        worker_reqs: List[WorkerRequest] = None,
        prompt: List[str] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = None,
        latents: Dict[str, torch.Tensor] = None,
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

        assert prompt is None and negative_prompt is None and num_inference_steps is None and latents is None, (
            "These parameters must be passed via worker_reqs.")
        assert worker_reqs is not None, "Worker requests must be passed!"

        # Extract params:
        prompt: List = []
        negative_prompt: List = []
        for req in worker_reqs:
            prompt.append(req.sampling_params.prompt)
            negative_prompt.append(req.sampling_params.negative_prompt)
        
        # 1. Check input
        self.check_inputs(
            prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
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
            self._cross_attention_kwargs.get("scale", None) if self._cross_attention_kwargs is not None else None
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
            clip_skip=clip_skip,
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
        self.scheduler.batch_set_timesteps(worker_reqs, device=device)

        # 5. Prepare latent variables
        # Latents must be kept separate, since it will be stored independently
        num_channels_latents = self.unet.config.in_channels
        latents: List = []
        for req in worker_reqs:
            if req.sampling_params.latents is None:
                latent = self.prepare_latents(
                    batch_size=1,
                    num_channels_latents=num_channels_latents,
                    height=req.sampling_params.height,
                    width=req.sampling_params.width,
                    dtype=prompt_embeds.dtype,
                    device=device,
                    generator=generator)
            else:
                base_shape = (1, num_channels_latents, req.sampling_params.height // self.vae_scale_factor, 
                              req.sampling_params.width // self.vae_scale_factor)
                latent = req.sampling_params.latents.reshape(base_shape).to(device)
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
            ).to(device=device, dtype=latents[0].dtype)
        
        # Store prepare result to requests
        for i, req in enumerate(worker_reqs):
            # update necessary variables
            req.sampling_params.latents = latents[i]
            req.sampling_params.prompt_embeds = prompt_embeds[i].unsqueeze(dim=0)
            req.sampling_params.negative_prompt_embeds = negative_prompt_embeds[i].unsqueeze(dim=0)
            # Create prepare output
            prepare_output = StableDiffusionEsymredPipelinePrepareOutput(
                timestep_cond=timestep_cond,
                added_cond_kwargs=added_cond_kwargs,
                extra_step_kwargs=extra_step_kwargs,
                device=device,
                do_classifier_free_guidance=self.do_classifier_free_guidance,)
            req.prepare_output = prepare_output

    
    @torch.inference_mode()
    def denoising_step(
        self,
        worker_reqs: Dict[str, List[WorkerRequest]],
        cross_attention_kwargs: Optional[Dict[str, Any]],
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]],
        callback_on_step_end_tensor_inputs: List[str],
        guidance_rescale: float,
        guidance_scale: float,
        timestep_cond: torch.Tensor,
        added_cond_kwargs: Optional[Dict],
        extra_step_kwargs: Dict,
        do_classifier_free_guidance: bool,
        is_sliced: bool,
        patch_size: int,
    ) -> None:
        # keep the iteration in fixed order
        resolution_list = list(worker_reqs.keys())
        resolution_list.sort(key= lambda res_str: int(res_str))
    
        latent_dict: Dict[str, torch.Tensor] = {}
        prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        negative_prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        timestep_dict: Dict[str, torch.Tensor] = {}

        for res in resolution_list:
            local_latent_list = []
            local_prompt_embeds_list = []
            local_negative_prompt_embeds_list = []
            local_timestep_list = []
            for wr in worker_reqs[res]:
                local_latent_list.append(wr.sampling_params.latents)
                local_prompt_embeds_list.append(wr.sampling_params.prompt_embeds)
                local_negative_prompt_embeds_list.append(wr.sampling_params.negative_prompt_embeds)
                local_timestep_list.append(wr.scheduler_states.get_next_timestep())
            latent_dict[res] = torch.cat(local_latent_list, dim=0)
            prompt_embeds_dict[res] = torch.cat(local_prompt_embeds_list, dim=0)
            negative_prompt_embeds_dict[res] = torch.cat(local_negative_prompt_embeds_list, dim=0)
            timestep_dict[res] = torch.tensor(data=local_timestep_list, dtype=wr.scheduler_states.timesteps.dtype,
                                              device=wr.scheduler_states.timesteps.device)
        
        latent_input_dict: Dict[str, torch.Tensor] = {}

        # classifier free
        if do_classifier_free_guidance:
            prompt_embeds_list: List[torch.Tensor] = []
            timestep_list: List[torch.Tensor] = []
            for res in resolution_list:
                latent_input_dict[res] = torch.cat([latent_dict[res]] * 2, dim=0)
                prompt_embeds_list.append(negative_prompt_embeds_dict[res])
                prompt_embeds_list.append(prompt_embeds_dict[res])
                timestep_list.extend([timestep_dict[res]] * 2)
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
            t = torch.cat(timestep_list, dim=0)
        else:
            prompt_embeds_list = []
            timestep_list = []
            for res in resolution_list:
                prompt_embeds_list.append(prompt_embeds_dict[res])
                timestep_list.append(timestep_dict[res])
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
            t = torch.cat(timestep_list, dim=0)
        
        # Scale input
        for res in resolution_list:
            latent_input_dict[res] = self.scheduler.batch_scale_model_input(worker_reqs=worker_reqs[res],
                                                                      samples=latent_input_dict[res],
                                                                      timestep_list=timestep_dict[res])

        noise_pred = self.unet(
            latent_input_dict,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            is_sliced=is_sliced,
            patch_size=patch_size
        )[0]

        for res, res_split_noise in noise_pred.items():
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = res_split_noise.chunk(2)
                res_split_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                res_split_noise = rescale_noise_cfg(res_split_noise, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latent_dict[res] = self.scheduler.batch_step(worker_reqs[res], res_split_noise, timestep_dict[res], 
                                                         latent_dict[res], **extra_step_kwargs, return_dict=False)
        
        # Update parameters
        for res in resolution_list:
            for i, req in enumerate(worker_reqs[res]):
                req.scheduler_states.update_states_one_step()
                req.sampling_params.latents = latent_dict[res][i].unsqueeze(dim=0)
    
    
    @torch.inference_mode()
    def post_inference(
        self,
        worker_reqs: Dict[int, List[WorkerRequest]],
        output_type: str,
        device: torch.device,
        prompt_embeds_dtype: torch.dtype,
        generator: torch.Generator,
    ) -> None:
        latent_dict: Dict[str, torch.Tensor] = {}
        for res in worker_reqs:
            latent_list = []
            for wr in worker_reqs[res]:
                latent_list.append(wr.sampling_params.latents)
            latent_dict[res] = torch.cat(latent_list, dim=0)

        for res, latent in latent_dict.items():
            image = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

            for i, req in enumerate(worker_reqs[res]):
                req.output = StableDiffusionEsymredPipelineOutput(
                    images=image[i],
                    nsfw_content_detected=None,)
        
    
    def check_inputs(
        self,
        prompt,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
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