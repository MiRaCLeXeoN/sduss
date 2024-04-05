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
class StableDiffusionPipelineSamplingParams(BaseSamplingParams):
    """Sampling parameters for StableDiffusionPipeline."""
    # Params that must be set as default
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
    callback_on_step_end_tensor_inputs: List[str] = field(default_factory=list)
    # Params that vary
    # Defined in BaseSampling Params

    
    def __post_init__(self):
        super().__post_init__()
        self.callback_on_step_end_tensor_inputs.append("latents")
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