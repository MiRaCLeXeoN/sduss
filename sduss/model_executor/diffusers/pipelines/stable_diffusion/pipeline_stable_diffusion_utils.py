from dataclasses import dataclass, fields, field
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
class StableDiffusionPipelinePrepareOutput:
    """Params that are same as sampling_params will not be stored here."""
    timestep_cond: torch.Tensor
    # latents: torch.Tensor  # update to sampling_params
    # prompt_embeds: torch.FloatTensor  # update to sampling_params
    added_cond_kwargs: Optional[Dict]
    extra_step_kwargs: Dict
    device: torch.device
    do_classifier_free_guidance: bool


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


class StableDiffusionPipelineSamplingParams(BaseSamplingParams):
    """Sampling parameters for StableDiffusionPipeline."""
    # Params that vary
    # Defined in BaseSampling Params
    volatile_params = {
        "timesteps" : None,
        "guidance_scale" : 7.5,
        "eta" : 0.0,
        "generator" : None,
        "ip_adapter_image" : None,
        "ip_adapter_image_embeds" : None,
        "output_type" : "pil",
        "return_dict" : True,
        "cross_attention_kwargs" : None,
        "guidance_rescale" : 0.0,
        "clip_skip" : None,
        "callback_on_step_end" : None,
        "callback_on_step_end_tensor_inputs" : ["latents"]
    }
    
    utils_cls = {
        "prepare_output" : StableDiffusionPipelinePrepareOutput,
        "step_input" : StableDiffusionPipelineStepInput,
        "step_output" : StableDiffusionPipelineStepOutput,
        "post_input" : StableDiffusionPipelinePostInput,
        "pipeline_output" : StableDiffusionPipelineOutput,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height: Optional[int] = self.resolution
        self.width: Optional[int] = self.resolution

        self.timesteps: List[int] = kwargs.pop("timesteps", None)
        self.guidance_scale: float = kwargs.pop("guidance_scale", 7.5)
        self.eta: float = kwargs.pop("eta", 0.0)
        self.generator: Optional[Union[torch.Generator, List[torch.Generator]]] = kwargs.pop("generator", None)
        self.ip_adapter_image: Optional[PipelineImageInput] = kwargs.pop("ip_adapter_image", None)
        self.ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = kwargs.pop("ip_adapter_image_embeds", None)
        self.output_type: Optional[str] = kwargs.pop("output_type", "pil")
        self.return_dict: bool = kwargs.pop("return_dict", True)
        self.cross_attention_kwargs: Optional[Dict[str, Any]] = kwargs.pop("cross_attention_kwargs", None)
        self.guidance_rescale: float = kwargs.pop("guidance_rescale", 0.0)
        self.clip_skip: Optional[int] = kwargs.pop("clip_skip", None)
        self.callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = kwargs.pop("callback_on_step_end", None)
        self.callback_on_step_end_tensor_inputs: List[str] = kwargs.pop("callback_on_step_end_tensor_inputs", ["latents"])
        self._check_volatile_params()

    
    def _check_volatile_params(self):
        """Check volatile params to ensure they are the same as default."""
        for param_name in self.volatile_params:
            if getattr(self, param_name) != self.volatile_params[param_name]:
                raise RuntimeError(f"Currently, we do not support customized {param_name} parameter.")