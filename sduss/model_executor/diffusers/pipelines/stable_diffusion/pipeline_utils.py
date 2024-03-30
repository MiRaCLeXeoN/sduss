from dataclasses import dataclass, fields
from typing import Union, Optional, List, Dict, Callable, Any, Type

import PIL
import numpy as np
import torch

from ..pipeline_utils import (BasePipelineStepInput, BasePipelinePostInput)

from sduss.model_executor.utils import BaseOutput
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.model_executor.diffusers.image_processor import PipelineImageInput
from sduss.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StableDiffusionPipelineSamplingParams(BaseSamplingParams):
    """Sampling parameters for StableDiffusionPipeline."""
    # Params that must be the same if to be batched
    height: Optional[int] = 512
    width: Optional[int] = 512
    guidance_scale: float = 7.5
    eta: float = 0.0
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    ip_adapter_image: Optional[PipelineImageInput] = None
    ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None
    output_type: Optional[str] = "pil"
    return_dict: bool = True
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    guidance_rescale: float = 0.0
    clip_skip: Optional[int] = None
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None
    callback_on_step_end_tensor_inputs: List[str] = ["latents"]
    # Params that can vary even when batched
    # Defined in BaseSampling Params

    
    def __post_init__(self):
        super().__post_init__()
        # Parameters that must be the same if to be batched
        self.volatile_params = {
            "height" : self.height,
            "width" : self.width,
            "guidance_scale" : self.guidance_scale,
            "eta" : self.eta,
            "generator" : self.generator,
            "ip_adapter_image" : self.ip_adapter_image,
            "ip_adapter_image_embeds" : self.ip_adapter_image_embeds,
            "output_type" : self.output_type,
            "return_dict" : self.return_dict,
            "cross_attention_kwargs" : self.cross_attention_kwargs,
            "guidance_rescale" : self.guidance_rescale,
            "clip_skip" : self.clip_skip,
            "callback_on_step_end" : self.callback_on_step_end,
            "callback_on_step_end_tensor_inputs" : self.callback_on_step_end_tensor_inputs 
        }
        
        self.utils_cls = {
            "prepare_output" : StableDiffusionPipelinePrepareOutput,
            "step_input" : StableDiffusionPipelineStepInput,
            "step_output" : StableDiffusionPipelineStepOutput,
            "post_input" : StableDiffusionPipelinePostInput,
            "pipeline_output" : StableDiffusionPipelineOutput,
        }


    def is_compatible_with(self, sp: "StableDiffusionPipelineSamplingParams") -> bool:
        for name in self.volatile_params.keys():
            if self.volatile_params[name] != sp.volatile_params[name]:
                return False
        return True
    
            

@dataclass
class StableDiffusionPipelinePrepareOutput:
    timesteps: List[int]
    timestep_cond: torch.Tensor
    latents: torch.Tensor
    prompt_embeds: torch.FloatTensor
    added_cond_kwargs: Optional[Dict]
    extra_step_kwargs: Dict
    device: torch.device
    # These are directly from `prepare_inference` args
    output_type: Optional[str]
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]
    return_dict: bool
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]]
    callback_on_step_end_tensor_inputs: List[str]
    do_classifier_free_guidance: bool
    guidance_rescale: float
    guidance_scale: float
    cross_attention_kwargs: Optional[Dict[str, Any]]

@dataclass
class StableDiffusionPipelineStepInput(BasePipelineStepInput):
    """Input wrapper for each denoising step.

    Args:
        timesteps (`List[int]`)
            List of timesteps to be processed. **Notice** that we might iterate multiple
            steps within each `denoise_step` call.
        num_inference_steps: (`int`)
            Number of total inference steps.
        timestep_cond: torch.Tensor
        latents: torch.Tensor
        prompt_embeds: torch.FloatTensor
        added_cond_kwargs: Optional[Dict]
        extra_step_kwargs: Dict
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]]
        callback_on_step_end_tensor_inputs: List[str]
    """
    timesteps: List[int]
    timestep_idxs: List[int]
    timestep_cond: torch.Tensor
    latents: torch.Tensor
    prompt_embeds: torch.FloatTensor
    added_cond_kwargs: Optional[Dict]
    extra_step_kwargs: Dict
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]]
    callback_on_step_end_tensor_inputs: List[str]
    do_classifier_free_guidance: bool
    guidance_rescale: float
    guidance_scale: float
    cross_attention_kwargs: Optional[Dict[str, Any]]
    
    def __post_init__(self):
        # Keep the original complete timesteps
        self._complete_timesteps = self.timesteps

    @classmethod
    def from_prepare_output(cls, output: StableDiffusionPipelinePrepareOutput
    ) -> "StableDiffusionPipelineStepInput": 
        return cls(timesteps=output.timesteps,
                   timestep_idxs=list(range(len(output.timesteps))),
                   timestep_cond=output.timestep_cond,
                   latents=output.latents,
                   prompt_embeds=output.prompt_embeds,
                   added_cond_kwargs=output.added_cond_kwargs,
                   extra_step_kwargs=output.extra_step_kwargs,
                   callback_on_step_end=output.callback_on_step_end,
                   callback_on_step_end_tensor_inputs=output.callback_on_step_end_tensor_inputs,
                   do_classifier_free_guidance=output.do_classifier_free_guidance,
                   guidance_rescale=output.guidance_rescale,
                   guidance_scale=output.guidance_scale,
                   cross_attention_kwargs=output.cross_attention_kwargs)
        
    def update_args(
        self,
        num_steps: int = 1,
        last_output: Optional["StableDiffusionPipelineStepOutput"] = None,
    ) -> "StableDiffusionPipelineStepInput":
        """Update input args with output from last step.
        
        Args:
            num_steps: (int)
                Number of steps to be processed for next round. This will influence the
                `timesteps` updated for next round.
            last_output: (StableDiffusionPipelineStepOutput)
                Output from last round
        """
        if last_output is None:
            # The first round
            self.timesteps = self._complete_timesteps[:num_steps]
            self.timestep_idxs = list(range(num_steps))
        else:
            last_idx = self.timestep_idxs[-1]
            self.timesteps = self._complete_timesteps[last_idx+1 : last_idx+1+num_steps]
            self.timestep_idxs = list(range(last_idx+1, last_idx+1+num_steps))
            # Update output's args
            self.latents = last_output.latents
            self.prompt_embeds = last_output.prompt_embeds

        return self


@dataclass
class StableDiffusionPipelineStepOutput:
    latents: torch.Tensor
    prompt_embeds: torch.Tensor


@dataclass
class StableDiffusionPipelinePostInput(BasePipelinePostInput):
    output_type: str
    latents: torch.Tensor
    device: torch.device
    prompt_embeds: torch.Tensor
    generator: torch.Generator
    return_dict: bool


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]