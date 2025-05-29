import torch
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, Type, TYPE_CHECKING

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline as DiffusersStableDiffusion3Pipeline,
    )

from ..pipeline_utils import BasePipeline
from ...image_processor import PipelineImageInput
from .pipeline_stable_diffusion_3_esymred_utils import (
    StableDiffusion3EsymredPipelinePrepareInput, StableDiffusion3EsymredPipelinePrepareOutput,
    StableDiffusion3EsymredPipelineStepInput, StableDiffusion3EsymredPipelineStepOutput,
    StableDiffusion3EsymredPipelinePostInput, StableDiffusion3EsymredPipelineOutput,
    StableDiffusion3EsymredPipelineSamplingParams)

if TYPE_CHECKING:
    from sduss.worker.runner.wrappers import RunnerRequestDictType, RunnerRequest


class ESyMReDStableDiffusion3Pipeline(DiffusersStableDiffusion3Pipeline, BasePipeline):
    SUPPORT_MIXED_PRECISION = True
    SUPPORT_RESOLUTIONS = [512, 768, 1024]

    @classmethod
    def instantiate_pipeline(cls, **kwargs):
        sub_modules: Dict = kwargs.pop("sub_modules", {})

        transformer = sub_modules.pop("transformer", None)
        assert transformer is not None

        # # Lazy import to avoid cuda extension building.
        from sduss.model_executor.modules.SD3Transformer import PatchSD3Transformer2DModel
        transformer = PatchSD3Transformer2DModel(transformer)
        sub_modules["transformer"] = transformer

        return cls(**sub_modules)
    
    @staticmethod
    def get_sampling_params_cls() -> Type[StableDiffusion3EsymredPipelineSamplingParams]:
        return StableDiffusion3EsymredPipelineSamplingParams

    
    def __post_init__(self):
        # TODO
        pass

        
    @torch.inference_mode()
    def prepare_inference(
        self,
        runner_reqs: "RunnerRequestDictType" = None,
        guidance_scale: float = 7.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        skip_layer_guidance_scale: float = 2.8,
    ) -> None:
        # Other parameters are removed

        resolution_list = list(runner_reqs.keys())
        resolution_list.sort()

        # 0. Collect args
        worker_reqs_list: "List[RunnerRequest]" = []
        prompt_list = []
        prompt_2_list = []
        prompt_3_list = []
        negative_prompt_list = []
        negative_prompt_2_list = []
        negative_prompt_3_list = []
        num_steps_list = []

        for res in resolution_list:
            for req in runner_reqs[res]:
                worker_reqs_list.append(req)
                prompt_list.append(req.sampling_params.prompt)
                prompt_2_list.append(req.sampling_params.prompt_2)
                prompt_3_list.append(req.sampling_params.prompt_3)
                negative_prompt_list.append(req.sampling_params.negative_prompt)
                negative_prompt_2_list.append(req.sampling_params.negative_prompt_2)
                negative_prompt_3_list.append(req.sampling_params.negative_prompt_3)
                num_steps_list.append(req.sampling_params.num_inference_steps)

        # 1. Check inputs. Raise error if not correct
        # ! Input checking is omitted here.
        # self.check_inputs(
        #     prompt,
        #     prompt_2,
        #     prompt_3,
        #     height,
        #     width,
        #     negative_prompt=negative_prompt,
        #     negative_prompt_2=negative_prompt_2,
        #     negative_prompt_3=negative_prompt_3,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     pooled_prompt_embeds=pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        #     max_sequence_length=max_sequence_length,
        # )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = len(prompt_list)
        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt_list,
            prompt_2=prompt_2_list,
            prompt_3=prompt_3_list,
            negative_prompt=negative_prompt_list,
            negative_prompt_2=negative_prompt_2_list,
            negative_prompt_3=negative_prompt_3_list,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # ! We should cat them in denoising step, not here
        # if self.do_classifier_free_guidance:
        #     # skip_guidance_layers will be set to None
        #     if skip_guidance_layers is not None:
        #         original_prompt_embeds = prompt_embeds
        #         original_pooled_prompt_embeds = pooled_prompt_embeds
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        #     pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        latent_list = []
        num_channels_latents = self.transformer.config.in_channels
        for res in resolution_list:
            for req in runner_reqs[res]:
                # We should ensure the consistency between different iteration
                # of the dict, so that list has the correct index
                if req.sampling_params.latents is None:
                    latent = self.prepare_latents(
                        1,
                        num_channels_latents,
                        height=req.sampling_params.height,
                        width=req.sampling_params.width,
                        dtype=prompt_embeds.dtype,
                        device=device,
                        generator=generator,
                        latents=None,
                    )
                else:
                    raise ValueError("Input latents are only expected to be None.")
                latent_list.append(latent)

        # 5. Prepare timesteps
        # scheduler_kwargs = {}
        # ! mu will be set to None, config is also None, so we jump over this part
        # if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        #     _, _, height, width = latents.shape
        #     image_seq_len = (height // self.transformer.config.patch_size) * (
        #         width // self.transformer.config.patch_size
        #     )
        #     mu = calculate_shift(
        #         image_seq_len,
        #         self.scheduler.config.base_image_seq_len,
        #         self.scheduler.config.max_image_seq_len,
        #         self.scheduler.config.base_shift,
        #         self.scheduler.config.max_shift,
        #     )
        #     scheduler_kwargs["mu"] = mu
        # elif mu is not None:
        #     scheduler_kwargs["mu"] = mu
        # ! sigmas and timesteps are all None. So retieve_timesteps can be simplified
        # timesteps, num_inference_steps = retrieve_timesteps(
        #     self.scheduler,
        #     num_inference_steps,
        #     device,
        #     sigmas=sigmas,
        #     **scheduler_kwargs,
        # )
        self.scheduler.batch_set_timesteps(worker_reqs_list, device=device)
        # num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        # ! Fixed to None, can be omitted
        # if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        #     ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
        #         ip_adapter_image,
        #         ip_adapter_image_embeds,
        #         device,
        #         batch_size * 1,
        #         self.do_classifier_free_guidance,
        #     )

        #     if self.joint_attention_kwargs is None:
        #         self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        #     else:
        #         self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)
        for i, req in enumerate(worker_reqs_list):
            req.sampling_params.prompt_embeds = prompt_embeds[i].unsqueeze(0)
            req.sampling_params.negative_prompt_embeds = negative_prompt_embeds[i].unsqueeze(0)
            req.sampling_params.latents = latent_list[i]
            # Create prepare output
            req.prepare_output = StableDiffusion3EsymredPipelinePrepareOutput(
                pooled_prompt_embeds=pooled_prompt_embeds[i].unsqueeze(0),  # (1, 1280)
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[i].unsqueeze(0),
                device=device,
                do_classifier_free_guidance=self.do_classifier_free_guidance)


    @torch.inference_mode()
    def denoising_step(
        self,
        runner_reqs: "Dict[str, List[RunnerRequest]]",
        do_classifier_free_guidance: bool = True,
        guidance_scale: float = 7.0,
        is_sliced: bool = False,
        patch_size: int = 256,
    ) -> None:
        # keep the iteration in fixed order
        resolution_list = list(runner_reqs.keys())
        resolution_list.sort(key= lambda res_str: int(res_str))
        # Collect args into dict
        latent_dict: Dict[str, torch.Tensor] = {}
        prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        negative_prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        pooled_prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        negative_pooled_prompt_embeds_dict: Dict[str, torch.Tensor] = {}
        timestep_dict: Dict[str, torch.Tensor] = {}
        input_index_dict: Dict[str, List[str]] = {}

        for res in resolution_list:
            local_latent_list = []
            local_prompt_embeds_list = []
            local_negative_prompt_embeds_list = []
            local_pooled_prompt_embeds_list = []
            local_negative_pooled_prompt_embeds_list = []
            local_timestep_list = []
            local_request_ids = []
            for req in runner_reqs[res]:
                local_request_ids.append(req.request_id)
                local_latent_list.append(req.sampling_params.latents)
                local_prompt_embeds_list.append(req.sampling_params.prompt_embeds)
                local_negative_prompt_embeds_list.append(req.sampling_params.negative_prompt_embeds)
                local_pooled_prompt_embeds_list.append(req.prepare_output.pooled_prompt_embeds)
                local_negative_pooled_prompt_embeds_list.append(req.prepare_output.negative_pooled_prompt_embeds)
                local_timestep_list.append(req.scheduler_states.get_next_timestep())
            latent_dict[res] = torch.cat(local_latent_list, dim=0)
            prompt_embeds_dict[res] = torch.cat(local_prompt_embeds_list, dim=0)
            negative_prompt_embeds_dict[res] = torch.cat(local_negative_prompt_embeds_list, dim=0)
            pooled_prompt_embeds_dict[res] = torch.cat(local_pooled_prompt_embeds_list, dim=0)
            negative_pooled_prompt_embeds_dict[res] = torch.cat(local_negative_pooled_prompt_embeds_list, dim=0)
            timestep_dict[res] = torch.tensor(data=local_timestep_list, dtype=req.scheduler_states.timesteps.dtype,
                                              device=req.scheduler_states.timesteps.device)
            input_index_dict[res] = local_request_ids
        # if self.interrupt:
        #     continue

        # We shoule preserve the lantent_dict as original, so we
        # use another dict
        latent_input_dict : Dict[str, torch.Tensor] = {}

        if do_classifier_free_guidance:
            prompt_embeds_list: List[torch.Tensor] = []
            pooled_prompt_embeds_list: List[torch.Tensor] = []
            timestep_list: List[torch.Tensor] = []
            for res in resolution_list:
                latent_input_dict[res] = torch.cat([latent_dict[res]] * 2, dim=0)
                input_index_dict[res] = [str(id) for id in input_index_dict[res]] + [f"{id}-1" for id in input_index_dict[res]]
                prompt_embeds_list.append(negative_prompt_embeds_dict[res])
                prompt_embeds_list.append(prompt_embeds_dict[res])
                pooled_prompt_embeds_list.append(negative_pooled_prompt_embeds_dict[res])
                pooled_prompt_embeds_list.append(pooled_prompt_embeds_dict[res])
                timestep_list.extend([timestep_dict[res]] * 2)
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)
            t = torch.cat(timestep_list, dim=0)
        else:
            prompt_embeds_list = []
            pooled_prompt_embeds_list: List[torch.Tensor] = []
            timestep_list = []
            for res in resolution_list:
                latent_input_dict[res] = latent_dict[res]
                prompt_embeds_list.append(prompt_embeds_dict[res])
                pooled_prompt_embeds_list.append(pooled_prompt_embeds_dict[res])
                timestep_list.append(timestep_dict[res])
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)
            t = torch.cat(timestep_list, dim=0)


        noise_pred = self.transformer(
            hidden_states=latent_input_dict,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
            is_sliced=is_sliced,
            patch_size=patch_size,
            input_indices=input_index_dict,
        )[0]

        # perform guidance
        for res, res_split_noise in noise_pred.items():
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = res_split_noise.chunk(2)
                res_split_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # ! skip_guidance_layers should be None, safe to omit
                # should_skip_layers = (
                #     True
                #     if i > num_inference_steps * skip_layer_guidance_start
                #     and i < num_inference_steps * skip_layer_guidance_stop
                #     else False
                # )
                # if skip_guidance_layers is not None and should_skip_layers:
                #     timestep = t.expand(latents.shape[0])
                #     latent_model_input = latents
                #     noise_pred_skip_layers = self.transformer(
                #         hidden_states=latent_model_input,
                #         timestep=timestep,
                #         encoder_hidden_states=original_prompt_embeds,
                #         pooled_projections=original_pooled_prompt_embeds,
                #         joint_attention_kwargs=self.joint_attention_kwargs,
                #         return_dict=False,
                #         skip_layers=skip_guidance_layers,
                #     )[0]
                #     noise_pred = (
                #         noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                #     )

            # Do scheduler step res by res, not in a whole
            # compute the previous noisy sample x_t -> x_t-1
            latent_dtype = latent_dict[res].dtype
            latent_dict[res] = self.scheduler.batch_step(runner_reqs=runner_reqs[res], 
                                                   model_outputs=res_split_noise, 
                                                   samples=latent_dict[res],
                                                   timesteps=timestep_dict[res], 
                                                   return_dict=False)

            if latent_dict[res].dtype != latent_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latent_dict[res] = latent_dict[res].to(latent_dtype)

        # if callback_on_step_end is not None:
        #     callback_kwargs = {}
        #     for k in callback_on_step_end_tensor_inputs:
        #         callback_kwargs[k] = locals()[k]
        #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

        #     latents = callback_outputs.pop("latents", latents)
        #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
        #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
        #     negative_pooled_prompt_embeds = callback_outputs.pop(
        #         "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
        #     )

        # call the callback, if provided
        # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
        #     progress_bar.update()

        # if XLA_AVAILABLE:
        #     xm.mark_step()
        for res in resolution_list:
            for i, req in enumerate(runner_reqs[res]):
                req.scheduler_states.update_states_one_step()
                req.sampling_params.latents = latent_dict[res][i].unsqueeze(dim=0)


    @torch.inference_mode()
    def post_inference(
        self,
        runner_reqs: "RunnerRequestDictType",
        output_type: str = "pil",
    ) -> None:
        for res in runner_reqs:
            # Group latents
            latent_list = []
            for req in runner_reqs[res]:
                latent_list.append(req.sampling_params.latents)
            latents = torch.cat(latent_list, dim=0)

            if output_type == "latent":
                raise ValueError("latent output is not supported!")
                image = latents
            else:
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
            
            for i, req in enumerate(runner_reqs[res]):
                req.output = StableDiffusion3EsymredPipelineOutput(
                    images=image[i],
                )